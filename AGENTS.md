# Repository Guidelines

We are building a simple renderer that loads Wavefront `.obj` models and writes rendered images. Follow this guide when adding code or opening pull requests.

## Project Structure & Module Organization
- `Cargo.toml`: crate metadata (Rust 2024).
- `src/main.rs`: entry point; keep it thin and delegate to modules.
- Suggested modules: `obj_loader/` (parsing vertices/indices/materials), `renderer/` (transform + raster), `camera/`, `math/`, `image/`.
- Put sample `.obj`/`.mtl` assets in `assets/`; generated frames go to `target/`.

## Build, Test, and Development Commands
- `cargo fmt` – format with `rustfmt`; run before commits.
- `cargo clippy --all-targets --all-features` – lint for correctness/style.
- `cargo test` – run unit/integration tests.
- `cargo run -- path/to/model.obj` – run the renderer locally.
- `cargo build --release` – optimized build in `target/release/`.

## Coding Style & Naming Conventions
- Use `rustfmt` defaults (4-space indent, grouped imports).
- Naming: `snake_case` for functions/vars, `PascalCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- Keep functions small; pass explicit types/config at boundaries.
- Document math, coordinate systems, and file-format assumptions with concise `///` docs.

## Testing Guidelines
- Co-locate unit tests with code via `#[cfg(test)] mod tests`.
- Integration tests live in `tests/` with names like `renders_cube_obj`; fixtures in `tests/fixtures/`.
- Prefer verifying dimensions/checksums over exact pixels unless using goldens.
- Run `cargo test` and `cargo clippy` before PRs; mention any fixtures added.

## Commit & Pull Request Guidelines
- Commits: short imperative subject (e.g., `Add obj face parser`); keep them buildable and focused.
- Avoid mixing formatting-only and feature changes in one commit.
- PRs: explain what/why, link issues, list checks run; attach screenshots or rendered outputs for visual changes.
- Request review after local checks (or CI) are green.

## Security & Configuration Tips
- Do not commit `target/`, OS files, or secrets; add ignores as needed.
- Add dependencies deliberately; run `cargo update` intentionally and audit when possible.
- Validate external inputs (`.obj`, CLI args) and prefer `Result` over panics across module boundaries.

## Architecture Notes
- Pipeline: load `.obj` into mesh structs, transform with camera/view/projection, rasterize triangles into a framebuffer, write PNG/PPM.
- Keep renderer pure where possible; separate loading, transformation, and output.
- Use config structs for camera FOV, image size, and background color instead of globals or scattered constants.
