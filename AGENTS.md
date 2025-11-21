# Repository Guidelines (keep this updated and concise after every change)

## Structure
- `src/main.rs`: thin entry; delegates to loader/window/renderer.
- `src/obj_loader/`: OBJ parsing; add renderer/camera/math/image modules as they appear.
- `src/window.rs`: winit setup + event loop driving renderer.
- `src/renderer.rs`: wgpu surface/device/queue config and frame rendering.
- Assets live in `assets/`; outputs go to `target/`.

## Commands (run after each change)
- `cargo fmt`; `cargo clippy --all-targets --all-features`; `cargo test`.
- `cargo run -- <model.obj>` to load a model; `cargo build --release` for optimized binaries.

## Style
- Rustfmt defaults; `snake_case` vars/fns, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants.
- Small functions, explicit boundaries, `Result` over panics; short `///` docs for math/format assumptions.

## Testing & PRs
- Unit tests co-located; integration in `tests/` with fixtures in `tests/fixtures/`.
- Before PR: fmt, clippy, test; commits are small/imperative, PRs explain why/what and checks run.

## Security
- No secrets/`target/`/OS cruft; deliberate dependency changes; validate external inputs.

## Architecture Snapshot
- Pipeline: load OBJ → transform (camera/view/projection) → rasterize triangles → write PNG/PPM or draw via wgpu.
- Config structs for camera FOV/image size/background; keep renderer logic pure.
- Windowed path: winit event loop + wgpu surface; redraw requests trigger render pass.

## Implementation Snapshot
- OBJ loader API (`src/obj_loader/mod.rs`): `load_obj(path) -> Result<ObjModel>`, with `ObjModel` holding `positions`, `texcoords`, `normals`, `faces`; `Face` uses `VertexIndex { position, texcoord, normal }`.
- Supports negative indices; faces with <3 refs are skipped.
- Renderer (`src/renderer.rs`): initializes wgpu instance/adapter/device/queue, configures surface with sRGB format and vsync-friendly present mode, builds depth buffer, and renders triangles with a simple WGSL shader using an MVP uniform. `upload_model` triangulates OBJ faces, computes normals (falling back to face normals), scales/translates to fit view, and pushes vertices into a GPU buffer. Orbit camera tracks yaw/pitch/distance; uniform is updated via `orbit_camera` (drag) and `zoom_camera` (wheel). Simple CPU bitmap UI draws an FPS counter in the top-left corner each frame. Includes a shader compilation/pipeline test to catch WGSL regressions.
- Event loop (`src/window.rs`): creates window, drives resize/close/redraw events, and requests redraws when idle; handles surface loss/outdated events by resizing. Mouse: left-drag orbits the camera (both axes inverted from initial), scroll wheel zooms; FPS is measured per second and pushed to renderer overlay. Keyboard: WASD rotates/zooms (A/D yaw, W forward zoom, S backward zoom).
- Entry (`src/main.rs`): spins up window + renderer, loads `assets/benz.obj` by default (CLI arg overrides), uploads it to renderer, then starts the event loop; leaks window to satisfy `'static` surface lifetime for long-lived event loop.
- Tests: unit tests in `obj_loader` including `assets/benz.obj` fixture (positions=332,922; texcoords=233,002; normals=195,844; faces=192,985 with one degenerate skipped) and a negative-index case.
