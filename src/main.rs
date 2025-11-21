use std::env;
use std::error::Error;
use std::process;

mod obj_loader;
mod renderer;
mod window;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = window::create_event_loop()?;
    let window = Box::leak(Box::new(window::create_window(&event_loop)?));
    let mut renderer = renderer::Renderer::new(window)?;

    let default_model = format!("{}/assets/benz.obj", env!("CARGO_MANIFEST_DIR"));
    let model_path = env::args().nth(1).unwrap_or(default_model);
    let model = obj_loader::load_obj(&model_path)?;
    renderer.upload_model(&model)?;

    println!(
        "Loaded {model_path} with {} vertices, {} texcoords, {} normals, {} faces",
        model.positions.len(),
        model.texcoords.len(),
        model.normals.len(),
        model.faces.len()
    );

    window::run_window(event_loop, window, renderer)?;
    Ok(()) // Event loop exits on close, so reaching here means the window was closed cleanly.
}
