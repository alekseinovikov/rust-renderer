use std::error::Error;

use glam::{Mat4, Vec3};
use wgpu::{SurfaceError, util::DeviceExt};
use winit::{dpi::PhysicalSize, window::Window};

use crate::obj_loader::ObjModel;
use crate::ui::UiOverlay;

pub struct Renderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    scene: SceneState,
    mesh: Option<MeshBuffers>,
    ui: UiOverlay,
}

impl<'window> Renderer<'window> {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    pub fn new(window: &'window Window) -> Result<Self, Box<dyn Error + 'window>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or("Failed to find a suitable GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("renderer_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| matches!(mode, wgpu::PresentMode::Fifo | wgpu::PresentMode::AutoVsync))
            .unwrap_or(surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_texture = Self::create_depth_texture(&device, &config);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let scene = SceneState::new(size);
        let globals = Globals {
            mvp: scene.mvp().to_cols_array_2d(),
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals_buffer"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let ui = UiOverlay::new(&device, surface_format, Self::DEPTH_FORMAT);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color: wgpu::Color {
                r: 0.07,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            },
            depth_texture,
            depth_view,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            scene,
            mesh: None,
            ui,
        })
    }

    pub fn upload_model(&mut self, model: &ObjModel) -> Result<(), Box<dyn Error>> {
        let (bounds_min, bounds_max) =
            compute_bounds(&model.positions).ok_or("Model has no positions")?;
        let center = (bounds_min + bounds_max) * 0.5;
        let mut radius = model
            .positions
            .iter()
            .map(|pos| {
                let p = Vec3::from_array(*pos);
                (p - center).length()
            })
            .fold(0.0, f32::max);
        if radius <= f32::EPSILON {
            radius = 1.0;
        }

        let model_transform =
            Mat4::from_scale(Vec3::splat(1.0 / radius)) * Mat4::from_translation(-center);
        self.scene.set_model_transform(model_transform);
        self.write_globals();

        let vertices = build_vertices(model);
        if vertices.is_empty() {
            return Err("Model has no drawable faces".into());
        }

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        self.mesh = Some(MeshBuffers {
            vertex_buffer,
            vertex_count: vertices
                .len()
                .try_into()
                .map_err(|_| "Too many vertices for GPU draw call")?,
        });

        Ok(())
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
        self.depth_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.scene.update_aspect(new_size);
        self.write_globals();
        self.ui.mark_dirty();
    }

    pub fn orbit_camera(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.scene.orbit(delta_yaw, delta_pitch);
        self.write_globals();
    }

    pub fn zoom_camera(&mut self, delta: f32) {
        self.scene.zoom(delta);
        self.write_globals();
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.ui.set_text(format!("FPS: {:.1}", fps));
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.ui.prepare(&self.device, self.size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let Some(mesh) = &self.mesh {
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.draw(0..mesh.vertex_count, 0..1);
            }

            self.ui.draw(&mut render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn write_globals(&mut self) {
        let globals = Globals {
            mvp: self.scene.mvp().to_cols_array_2d(),
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&globals));
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }
}

fn build_vertices(model: &ObjModel) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    for face in &model.faces {
        if face.vertices.len() < 3 {
            continue;
        }

        for tri_index in 1..face.vertices.len() - 1 {
            let a = &face.vertices[0];
            let b = &face.vertices[tri_index];
            let c = &face.vertices[tri_index + 1];

            let a_pos = Vec3::from_array(model.positions[a.position]);
            let b_pos = Vec3::from_array(model.positions[b.position]);
            let c_pos = Vec3::from_array(model.positions[c.position]);

            let face_normal = (b_pos - a_pos)
                .cross(c_pos - a_pos)
                .try_normalize()
                .unwrap_or(Vec3::Y);

            let provided_normals = [a, b, c].map(|vertex| {
                vertex
                    .normal
                    .and_then(|idx| model.normals.get(idx))
                    .copied()
                    .map(Vec3::from_array)
            });

            let normals = if provided_normals.iter().all(Option::is_some) {
                provided_normals
                    .map(|normal| normal.unwrap().try_normalize().unwrap_or(face_normal))
            } else {
                [face_normal; 3]
            };

            vertices.push(Vertex {
                position: a_pos.to_array(),
                normal: normals[0].to_array(),
            });
            vertices.push(Vertex {
                position: b_pos.to_array(),
                normal: normals[1].to_array(),
            });
            vertices.push(Vertex {
                position: c_pos.to_array(),
                normal: normals[2].to_array(),
            });
        }
    }

    vertices
}

fn compute_bounds(positions: &[[f32; 3]]) -> Option<(Vec3, Vec3)> {
    let mut iter = positions.iter();
    let first = Vec3::from_array(*iter.next()?);
    let mut min = first;
    let mut max = first;

    for pos in iter {
        let p = Vec3::from_array(*pos);
        min = min.min(p);
        max = max.max(p);
    }

    Some((min, max))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}

struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

struct SceneState {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl SceneState {
    fn new(size: PhysicalSize<u32>) -> Self {
        let aspect = aspect_ratio(size);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh_gl(60.0_f32.to_radians(), aspect, 0.1, 100.0);

        Self {
            model: Mat4::IDENTITY,
            view,
            projection,
            yaw: 0.0,
            pitch: 0.0,
            distance: 3.0,
        }
    }

    fn update_aspect(&mut self, size: PhysicalSize<u32>) {
        self.projection =
            Mat4::perspective_rh_gl(60.0_f32.to_radians(), aspect_ratio(size), 0.1, 100.0);
    }

    fn set_model_transform(&mut self, model: Mat4) {
        self.model = model;
    }

    fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.5, 1.5);
        self.update_view();
    }

    fn zoom(&mut self, delta: f32) {
        let zoom_amount = delta * 0.25;
        self.distance = (self.distance - zoom_amount).clamp(0.4, 20.0);
        self.update_view();
    }

    fn update_view(&mut self) {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let x = self.distance * cos_pitch * sin_yaw;
        let y = self.distance * sin_pitch;
        let z = self.distance * cos_pitch * cos_yaw;
        let eye = Vec3::new(x, y, z);
        self.view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    }

    fn mvp(&self) -> Mat4 {
        self.projection * self.view * self.model
    }
}

fn aspect_ratio(size: PhysicalSize<u32>) -> f32 {
    let width = size.width.max(1) as f32;
    let height = size.height.max(1) as f32;
    width / height
}

const SHADER_SOURCE: &str = r#"
struct Globals {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = globals.mvp * vec4<f32>(model.position, 1.0);
    out.normal = model.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lit = normalize(in.normal) * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    let color = vec3<f32>(0.1, 0.6, 0.9) * lit;
    return vec4<f32>(color, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_compiles_into_render_pipeline() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping shader test: no suitable adapter available");
            return;
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_globals_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Building the render pipeline ensures both vertex and fragment stages validate.
        let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Renderer::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .ok()
    }
}
