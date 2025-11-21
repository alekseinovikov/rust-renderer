use std::time::{Duration, Instant};

use wgpu::SurfaceError;
use winit::{
    error::{EventLoopError, OsError},
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use crate::renderer::Renderer;

/// Create an event loop for handling window events.
pub fn create_event_loop() -> Result<EventLoop<()>, EventLoopError> {
    EventLoop::new()
}

/// Build the main application window.
pub fn create_window(event_loop: &EventLoop<()>) -> Result<Window, OsError> {
    WindowBuilder::new()
        .with_title("rust-renderer")
        .build(event_loop)
}

/// Drive the winit event loop until the window is closed.
pub fn run_window(
    event_loop: EventLoop<()>,
    window: &'static Window,
    mut renderer: Renderer<'static>,
) -> Result<(), EventLoopError> {
    let mut input = InputState::default();
    let mut fps = FpsCounter::new();

    event_loop.run(move |event, event_loop_window_target| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => event_loop_window_target.exit(),
        Event::WindowEvent {
            event: WindowEvent::Resized(new_size),
            ..
        } => {
            renderer.resize(new_size);
            window.request_redraw();
        }
        Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                },
            ..
        } => {
            input.dragging = state == ElementState::Pressed;
            if !input.dragging {
                input.last_cursor = None;
            }
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            let position = (position.x as f32, position.y as f32);
            if input.dragging
                && let Some((last_x, last_y)) = input.last_cursor
            {
                let delta_x = position.0 - last_x;
                let delta_y = position.1 - last_y;
                renderer.orbit_camera(
                    -delta_x * input.rotate_sensitivity,
                    delta_y * input.rotate_sensitivity,
                );
                window.request_redraw();
            }
            input.last_cursor = Some(position);
        }
        Event::WindowEvent {
            event: WindowEvent::MouseWheel { delta, .. },
            ..
        } => {
            let scroll = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
            };
            if scroll.abs() > f32::EPSILON {
                renderer.zoom_camera(-scroll);
                window.request_redraw();
            }
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { event, .. },
            ..
        } if event.state == ElementState::Pressed => {
            if let PhysicalKey::Code(code) = event.physical_key {
                match code {
                    KeyCode::KeyA => renderer.orbit_camera(-input.key_rotate_step, 0.0),
                    KeyCode::KeyD => renderer.orbit_camera(input.key_rotate_step, 0.0),
                    KeyCode::KeyW => renderer.zoom_camera(input.key_zoom_step),
                    KeyCode::KeyS => renderer.zoom_camera(-input.key_zoom_step),
                    _ => return,
                }
                window.request_redraw();
            }
        }
        Event::AboutToWait => window.request_redraw(),
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => match renderer.render() {
            Ok(()) => {
                if let Some(value) = fps.on_frame() {
                    renderer.set_fps(value);
                }
            }
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                renderer.resize(window.inner_size())
            }
            Err(SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
            Err(SurfaceError::Timeout) => eprintln!("Surface timeout; continuing"),
        },
        _ => {}
    })
}

struct InputState {
    dragging: bool,
    last_cursor: Option<(f32, f32)>,
    rotate_sensitivity: f32,
    key_rotate_step: f32,
    key_zoom_step: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            dragging: false,
            last_cursor: None,
            rotate_sensitivity: 0.005,
            key_rotate_step: 0.08,
            key_zoom_step: 1.0,
        }
    }
}

struct FpsCounter {
    last: Instant,
    frames: u32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            frames: 0,
        }
    }

    fn on_frame(&mut self) -> Option<f32> {
        self.frames += 1;
        let now = Instant::now();
        let elapsed = now - self.last;
        if elapsed >= Duration::from_secs(1) {
            let fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last = now;
            Some(fps)
        } else {
            None
        }
    }
}
