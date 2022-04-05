mod piece;
mod render;
mod spline;
mod util;

use clap::Parser;
use image::open;
use render::PuzzleCreationOptions;
use std::time::Instant;

// Generated jigsaw puzzle pieces from an input image
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to input image
    #[clap(short, long)]
    image: String,

    /// File to output to
    #[clap(short, long)]
    output: String,

    /// Number of horizontal pieces
    #[clap(short)]
    x: u32,

    /// Number of vertical pieces
    #[clap(short)]
    y: u32,
}

async fn run(args: Args) {
    let src_image = open(args.image).unwrap().to_rgba8();

    if let Some(output) = render::render_puzzle(PuzzleCreationOptions {
        image: src_image,
        pieces_x: args.x,
        pieces_y: args.y,
    })
    .await
    {
        let saving_start = Instant::now();
        println!("Saving image...");
        let image = image::RgbaImage::from_raw(
            output.output_image_width,
            output.output_image_height,
            output.buffer,
        )
        .unwrap();

        image.save(args.output).unwrap();

        println!(
            "Took {}ms to save image",
            saving_start.elapsed().as_millis()
        );
    }
}

fn main() {
    let args = Args::parse();

    let start = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(args));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }

    println!("\nFinished in {}ms.", start.elapsed().as_millis());
}
