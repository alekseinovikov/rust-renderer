use wgpu::util::DeviceExt;
use wgpu::{RenderPipeline, TextureFormat};
use winit::dpi::PhysicalSize;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl UiVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}

pub struct UiOverlay {
    pipeline: RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_count: u32,
    text: String,
    dirty: bool,
    color: [f32; 3],
}

impl UiOverlay {
    const CELL_SIZE: f32 = 8.0;
    const ORIGIN: (f32, f32) = (10.0, 10.0);

    pub fn new(device: &wgpu::Device, format: TextureFormat, depth_format: TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui_shader"),
            source: wgpu::ShaderSource::Wgsl(UI_SHADER_SOURCE.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[UiVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            vertex_buffer: None,
            vertex_count: 0,
            text: "FPS: --".to_string(),
            dirty: true,
            color: [1.0, 1.0, 1.0],
        }
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if self.text != text {
            self.text = text;
            self.dirty = true;
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn prepare(&mut self, device: &wgpu::Device, size: PhysicalSize<u32>) {
        if !self.dirty {
            return;
        }

        if size.width == 0 || size.height == 0 {
            self.vertex_buffer = None;
            self.vertex_count = 0;
            self.dirty = false;
            return;
        }

        let vertices =
            build_text_vertices(&self.text, size, Self::ORIGIN, Self::CELL_SIZE, self.color);

        if vertices.is_empty() {
            self.vertex_buffer = None;
            self.vertex_count = 0;
        } else {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui_vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            self.vertex_count = vertices.len() as u32;
            self.vertex_buffer = Some(buffer);
        }

        self.dirty = false;
    }

    pub fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        if let Some(buffer) = &self.vertex_buffer {
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }
    }
}

fn build_text_vertices(
    text: &str,
    size: PhysicalSize<u32>,
    origin: (f32, f32),
    cell_size: f32,
    color: [f32; 3],
) -> Vec<UiVertex> {
    let mut vertices = Vec::new();
    let (mut cursor_x, cursor_y) = origin;
    let (width, height) = (size.width as f32, size.height as f32);

    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += cell_size;
            continue;
        }

        if let Some(bitmap) = glyph_bitmap(ch) {
            for (row_idx, row) in bitmap.iter().enumerate() {
                for (col_idx, c) in row.chars().enumerate() {
                    if c == '#' {
                        let x = cursor_x + col_idx as f32 * cell_size;
                        let y = cursor_y + row_idx as f32 * cell_size;
                        push_quad(
                            &mut vertices,
                            x,
                            y,
                            cell_size,
                            cell_size,
                            (width, height),
                            color,
                        );
                    }
                }
            }
            cursor_x += (bitmap[0].len() as f32 + 1.0) * cell_size;
        }
    }

    vertices
}

fn push_quad(
    vertices: &mut Vec<UiVertex>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    screen_size: (f32, f32),
    color: [f32; 3],
) {
    let (screen_w, screen_h) = screen_size;
    let x0 = (x / screen_w) * 2.0 - 1.0;
    let x1 = ((x + w) / screen_w) * 2.0 - 1.0;
    let y0 = 1.0 - (y / screen_h) * 2.0;
    let y1 = 1.0 - ((y + h) / screen_h) * 2.0;

    let v0 = UiVertex {
        position: [x0, y0],
        color,
    };
    let v1 = UiVertex {
        position: [x1, y0],
        color,
    };
    let v2 = UiVertex {
        position: [x1, y1],
        color,
    };
    let v3 = UiVertex {
        position: [x0, y1],
        color,
    };

    vertices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn glyph_bitmap(ch: char) -> Option<&'static [&'static str]> {
    match ch {
        '0' => Some(&["###", "# #", "# #", "# #", "###"]),
        '1' => Some(&["  #", "  #", "  #", "  #", "  #"]),
        '2' => Some(&["###", "  #", "###", "#  ", "###"]),
        '3' => Some(&["###", "  #", "###", "  #", "###"]),
        '4' => Some(&["# #", "# #", "###", "  #", "  #"]),
        '5' => Some(&["###", "#  ", "###", "  #", "###"]),
        '6' => Some(&["###", "#  ", "###", "# #", "###"]),
        '7' => Some(&["###", "  #", "  #", "  #", "  #"]),
        '8' => Some(&["###", "# #", "###", "# #", "###"]),
        '9' => Some(&["###", "# #", "###", "  #", "###"]),
        ':' => Some(&["   ", " # ", "   ", " # ", "   "]),
        '.' => Some(&["   ", "   ", "   ", "   ", " # "]),
        'F' => Some(&["###", "#  ", "###", "#  ", "#  "]),
        'P' => Some(&["## ", "# #", "## ", "#  ", "#  "]),
        'S' => Some(&[" ###", "#   ", " ##", "   #", "### "]),
        '-' => Some(&["   ", "   ", "###", "   ", "   "]),
        _ => None,
    }
}

const UI_SHADER_SOURCE: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
