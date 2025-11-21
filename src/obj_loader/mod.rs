use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Result type for OBJ parsing.
pub type Result<T> = std::result::Result<T, ObjError>;

/// A parsed Wavefront OBJ model.
pub struct ObjModel {
    pub positions: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub normals: Vec<[f32; 3]>,
    pub faces: Vec<Face>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexIndex {
    pub position: usize,
    pub texcoord: Option<usize>,
    pub normal: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Face {
    pub vertices: Vec<VertexIndex>,
}

#[derive(Debug)]
pub enum ObjError {
    Io(std::io::Error),
    Parse { line: usize, message: String },
}

impl Display for ObjError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjError::Io(err) => write!(f, "I/O error: {err}"),
            ObjError::Parse { line, message } => write!(f, "Parse error on line {line}: {message}"),
        }
    }
}

impl std::error::Error for ObjError {}

impl From<std::io::Error> for ObjError {
    fn from(value: std::io::Error) -> Self {
        ObjError::Io(value)
    }
}

pub fn load_obj(path: impl AsRef<Path>) -> Result<ObjModel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line_number = line_number + 1;
        let content = line?;
        let trimmed = content.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let Some(kind) = parts.next() else {
            continue;
        };

        match kind {
            "v" => {
                let coords: Vec<f32> = parts
                    .map(|value| parse_f32(value, "vertex coordinate", line_number))
                    .collect::<Result<_>>()?;

                if coords.len() < 3 {
                    return Err(ObjError::Parse {
                        line: line_number,
                        message: "vertex position requires at least 3 components".to_string(),
                    });
                }

                positions.push([coords[0], coords[1], coords[2]]);
            }
            "vt" => {
                let coords: Vec<f32> = parts
                    .map(|value| parse_f32(value, "texture coordinate", line_number))
                    .collect::<Result<_>>()?;

                if coords.len() < 2 {
                    return Err(ObjError::Parse {
                        line: line_number,
                        message: "texture coordinate requires at least 2 components".to_string(),
                    });
                }

                texcoords.push([coords[0], coords[1]]);
            }
            "vn" => {
                let coords: Vec<f32> = parts
                    .map(|value| parse_f32(value, "normal component", line_number))
                    .collect::<Result<_>>()?;

                if coords.len() < 3 {
                    return Err(ObjError::Parse {
                        line: line_number,
                        message: "normal requires 3 components".to_string(),
                    });
                }

                normals.push([coords[0], coords[1], coords[2]]);
            }
            "f" => {
                let mut vertices = Vec::new();

                for vertex in parts {
                    let vertex = parse_face_vertex(
                        vertex,
                        positions.len(),
                        texcoords.len(),
                        normals.len(),
                        line_number,
                    )?;
                    vertices.push(vertex);
                }

                if vertices.len() < 3 {
                    // Some files contain degenerate entries; skip them instead of failing the load.
                    continue;
                }

                faces.push(Face { vertices });
            }
            _ => {
                // Ignore statements we do not yet support (e.g. mtllib, usemtl, s).
            }
        }
    }

    Ok(ObjModel {
        positions,
        texcoords,
        normals,
        faces,
    })
}

fn parse_f32(value: &str, label: &str, line_number: usize) -> Result<f32> {
    value.parse::<f32>().map_err(|err| ObjError::Parse {
        line: line_number,
        message: format!("invalid {label} '{value}': {err}"),
    })
}

fn parse_index(value: &str, len: usize, label: &str, line_number: usize) -> Result<usize> {
    let raw = value.parse::<isize>().map_err(|err| ObjError::Parse {
        line: line_number,
        message: format!("invalid {label} index '{value}': {err}"),
    })?;

    if raw == 0 {
        return Err(ObjError::Parse {
            line: line_number,
            message: format!("{label} indices are 1-based in OBJ files"),
        });
    }

    let len = len as isize;
    let resolved = if raw > 0 { raw - 1 } else { len + raw };

    if resolved < 0 || resolved >= len {
        return Err(ObjError::Parse {
            line: line_number,
            message: format!("out of range {label} index {} (available: {len})", raw),
        });
    }

    Ok(resolved as usize)
}

fn parse_face_vertex(
    vertex: &str,
    position_count: usize,
    texcoord_count: usize,
    normal_count: usize,
    line_number: usize,
) -> Result<VertexIndex> {
    let mut parts = vertex.split('/');

    let position_str = parts.next().unwrap_or_default();
    if position_str.is_empty() {
        return Err(ObjError::Parse {
            line: line_number,
            message: "face vertex is missing a position index".to_string(),
        });
    }

    let position = parse_index(position_str, position_count, "position", line_number)?;

    let texcoord = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| parse_index(value, texcoord_count, "texture", line_number))
        .transpose()?;

    let normal = parts
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| parse_index(value, normal_count, "normal", line_number))
        .transpose()?;

    Ok(VertexIndex {
        position,
        texcoord,
        normal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_path(name: &str) -> String {
        let root = env!("CARGO_MANIFEST_DIR");
        format!("{root}/assets/{name}")
    }

    #[test]
    fn loads_benz_obj() {
        let model = load_obj(asset_path("benz.obj")).expect("failed to load benz.obj");

        assert_eq!(model.positions.len(), 332_922);
        assert_eq!(model.texcoords.len(), 233_002);
        assert_eq!(model.normals.len(), 195_844);
        assert_eq!(model.faces.len(), 192_985);

        let first_vertex = model.positions.first().expect("no positions parsed");
        assert!((first_vertex[0] - 0.943686).abs() < 1e-6);
        assert!((first_vertex[1] - 1.181801).abs() < 1e-6);
        assert!((first_vertex[2] - -3.548299).abs() < 1e-6);

        let first_face = model.faces.first().expect("no faces parsed");
        assert_eq!(first_face.vertices.len(), 3);
        assert_eq!(
            first_face.vertices[0],
            VertexIndex {
                position: 12,
                texcoord: Some(0),
                normal: Some(0),
            }
        );
        assert_eq!(
            first_face.vertices[1],
            VertexIndex {
                position: 45,
                texcoord: Some(1),
                normal: Some(0),
            }
        );
        assert_eq!(
            first_face.vertices[2],
            VertexIndex {
                position: 27,
                texcoord: Some(2),
                normal: Some(0),
            }
        );
    }

    #[test]
    fn supports_negative_indices() {
        let source = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vn 0.0 0.0 1.0
f -3/-3/-1 -2/-2/-1 -1/-1/-1
";

        let reader = BufReader::new(source.as_bytes());
        let mut positions = Vec::new();
        let mut texcoords = Vec::new();
        let mut normals = Vec::new();
        let mut faces = Vec::new();

        for (line_number, line) in reader.lines().enumerate() {
            let line_number = line_number + 1;
            let content = line.unwrap();
            let trimmed = content.trim();
            if trimmed.is_empty() {
                continue;
            }

            let mut parts = trimmed.split_whitespace();
            let kind = parts.next().unwrap();
            match kind {
                "v" => positions.push(read_vector3(parts, line_number)),
                "vt" => texcoords.push(read_vector2(parts, line_number)),
                "vn" => normals.push(read_vector3(parts, line_number)),
                "f" => {
                    let mut vertices = Vec::new();
                    for vertex in parts {
                        vertices.push(
                            parse_face_vertex(
                                vertex,
                                positions.len(),
                                texcoords.len(),
                                normals.len(),
                                line_number,
                            )
                            .unwrap(),
                        );
                    }
                    faces.push(Face { vertices });
                }
                _ => {}
            }
        }

        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        assert_eq!(
            face.vertices[0],
            VertexIndex {
                position: 0,
                texcoord: Some(0),
                normal: Some(0),
            }
        );
        assert_eq!(
            face.vertices[1],
            VertexIndex {
                position: 1,
                texcoord: Some(1),
                normal: Some(0),
            }
        );
        assert_eq!(
            face.vertices[2],
            VertexIndex {
                position: 2,
                texcoord: Some(2),
                normal: Some(0),
            }
        );
    }

    fn read_vector3<'a>(mut values: impl Iterator<Item = &'a str>, line_number: usize) -> [f32; 3] {
        [
            parse_f32(values.next().unwrap(), "vector", line_number).unwrap(),
            parse_f32(values.next().unwrap(), "vector", line_number).unwrap(),
            parse_f32(values.next().unwrap(), "vector", line_number).unwrap(),
        ]
    }

    fn read_vector2<'a>(mut values: impl Iterator<Item = &'a str>, line_number: usize) -> [f32; 2] {
        [
            parse_f32(values.next().unwrap(), "vector", line_number).unwrap(),
            parse_f32(values.next().unwrap(), "vector", line_number).unwrap(),
        ]
    }
}
