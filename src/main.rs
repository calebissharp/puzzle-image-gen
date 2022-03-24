mod piece;
mod polygon;
mod spline;
use crate::spline::CatmullRomSpline;

use clap::Parser;
use humantime::format_duration;
use image::error::{DecodingError, ImageError};
use image::imageops::{resize, FilterType};
use image::io::Reader as ImageReader;
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageBuffer, Pixel, RgbImage, Rgba, RgbaImage,
};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_cross_mut, draw_line_segment_mut, Blend,
};
use imageproc::pixelops::interpolate;
use nalgebra::{
    DMatrix, DVector, Matrix4, Matrix4x2, RowDVector, RowVector2, RowVector4, Vector2, Vector4,
};
use piece::MAX_JOINER_SIZE;
use rayon::prelude::*;
use std::vec::Vec;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to source image
    #[clap(short, long)]
    path: String,
    /// Output directory to store generated piece images
    #[clap(short, long)]
    output: String,
    /// Target size for the longest side
    #[clap(long, default_value_t = 4096)]
    target_resolution: u32,
}

fn slice_image(
    img: &RgbaImage,
    pieces_x: u32,
    pieces_y: u32,
    padding: u32,
) -> Result<Vec<RgbaImage>, ImageError> {
    let padded_img = add_padding_to_img(img, padding)?;

    let piece_height = img.height() / pieces_y;
    let piece_width = img.width() / pieces_x;

    let mut new_images = vec![
        RgbaImage::new(piece_width, piece_height);
        usize::try_from(pieces_x * pieces_y).unwrap()
    ];

    for y in 0..pieces_y {
        for x in 0..pieces_x {
            let i = usize::try_from(y * pieces_x + x).unwrap();

            new_images[i] = padded_img
                .view(
                    x * piece_width,
                    y * piece_height,
                    piece_width + padding * 2,
                    piece_height + padding * 2,
                )
                .to_image();
        }
    }

    Ok(new_images)
}

fn add_padding_to_img(img: &RgbaImage, padding: u32) -> Result<RgbaImage, ImageError> {
    let mut new_img = RgbaImage::new(img.width() + padding * 2, img.height() + padding * 2);

    new_img.copy_from(img, padding, padding)?;

    Ok(new_img)
}

// fn draw_grid(img: &mut RgbaImage) -> Result<(), ImageError> {
//     // img.put_pixel(x: u32, y: u32, pixel: P)
//     // let control_points = vec![
//     //     Key::new(0., [0., 0.], Interpolation::CatmullRom),
//     //     Key::new(5., [5., 5.], Interpolation::CatmullRom),
//     //     Key::new(10., [0., 0.], Interpolation::CatmullRom),
//     // ];
//     // let spline = Spline::from_vec(control_points);

//     // for i in 0..10 {
//     //     println!("{:?}", spline.sample(0.));
//     // }

//     let red = Rgba([255u8, 0, 0, 255u8]);

//     // draw_line_segment_mut(img, (5., 5.), (500., 500.), red);
//     draw_antialiased_line_segment_mut(img, (5, 5), (500, 500), red, |left, right, weight| {
//         interpolate(left, right, 1.1)
//     });

//     Ok(())
// }

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let mut img = ImageReader::open(&args.path)?
        .decode()
        .unwrap()
        .into_rgba8();

    println!(
        "Loaded image {}, ({}x{})",
        args.path,
        img.width(),
        img.height(),
    );

    let src_width = img.width();
    let src_height = img.height();

    let scale = args.target_resolution / std::cmp::max(src_width, src_height);

    let target_width = scale * src_width;
    let target_height = scale * src_height;

    println!("Resizing image to {}x{}", target_width, target_height);

    img = resize(&img, target_width, target_height, FilterType::CatmullRom);

    let padding = MAX_JOINER_SIZE;

    let pieces_x = 16;
    let pieces_y = 16;

    let mut pieces = slice_image(&img, pieces_x, pieces_y, padding).unwrap();

    let white = Rgba([255, 255, 255, 255]);
    let red = Rgba([255, 0, 0, 255]);
    let green = Rgba([0, 255, 0, 255]);
    let blue = Rgba([0, 0, 255, 255]);

    let piece_width = target_width / pieces_x;
    let piece_height = target_height / pieces_y;

    let top_points = piece::Edge::gen_edge(piece::Side::TOP, piece_width);
    let mut right_points = piece::Edge::gen_edge(piece::Side::RIGHT, piece_height);
    let mut bottom_points = piece::Edge::gen_edge(piece::Side::BOTTOM, piece_width);
    let mut left_points = piece::Edge::gen_edge(piece::Side::LEFT, piece_height);

    let mut control_points = top_points.points;
    control_points.append(&mut right_points.points);
    control_points.append(&mut bottom_points.points);
    control_points.append(&mut left_points.points);
    // control_points.push(control_points[0]);

    let spline = CatmullRomSpline::new(control_points.clone(), 0.5, true);

    let step = 10;

    let mut points = (0..(spline.control_points.len() - 1) * step)
        .map(|x| spline.sample(x as f64 / step as f64).unwrap().transpose())
        .collect::<Vec<Vector2<f64>>>();

    points.push(points[0]);

    let polygon = polygon::Polygon::new(points.clone());

    // img.save("test-image.png").unwrap();

    println!("Split into {} images", pieces.len());

    let mut points_iter = points.windows(2);

    let mut border_img = RgbaImage::new(288, 288);

    // while let Some(points) = points_iter.next() {
    //     let point = points[0];
    //     let next_point = points[1];
    //     // let mut color = white.clone();
    //     // color[3] = (point[0].fract() * 255.).round() as u8;

    //     // img.put_pixel(point[0].round() as u32, point[1].round() as u32, white);
    //     draw_antialiased_line_segment_mut(
    //         &mut pieces[18],
    //         (point[0].round() as i32, point[1].round() as i32),
    //         (next_point[0].round() as i32, next_point[1].round() as i32),
    //         white,
    //         interpolate,
    //     )
    // }

    let start = std::time::Instant::now();

    pieces.par_iter_mut().for_each(|piece| {
        piece.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
            if !polygon.check(x as f64, y as f64) {
                pixel[3] = 0;
            }
        });
        // for (x, y, pixel) in piece.enumerate_pixels_mut() {
        //     if !polygon.check(x as f64, y as f64) {
        //         pixel[3] = 0;
        //     }
        // }
    });

    let elapsed = start.elapsed();

    println!("Finished masking pieces (took {}ms)", elapsed.as_millis());

    // for (x, y, pixel) in pieces[18].enumerate_pixels_mut() {
    //     // let border_pixel = border_img.get_pixel(x, y);
    //     // if border_pixel[3] > 0 {
    //     //todo: antialias?
    //     // pixel[3] = border_pixel[3];
    //     if !polygon.check(x as f64, y as f64) {
    //         pixel[3] = 0;
    //     }
    // }

    while let Some(points) = points_iter.next() {
        let point = points[0];
        let next_point = points[1];
        // let mut color = white.clone();
        // color[3] = (point[0].fract() * 255.).round() as u8;

        // img.put_pixel(point[0].round() as u32, point[1].round() as u32, white);
        // draw_antialiased_line_segment_mut(
        //     &mut pieces[18],
        //     (point[0].round() as i32, point[1].round() as i32),
        //     (next_point[0].round() as i32, next_point[1].round() as i32),
        //     white,
        //     interpolate,
        // )
        // draw_line_segment_mut(
        //     &mut pieces[18],
        //     (point[0] as f32, point[1] as f32),
        //     (next_point[0] as f32, next_point[1] as f32),
        //     white,
        // )
    }

    for (i, control_point) in control_points.iter().enumerate() {
        assert_eq!(spline.sample(i as f64).unwrap().as_slice(), control_point);

        draw_cross_mut(
            &mut pieces[18],
            green,
            control_point[0].round() as i32,
            control_point[1].round() as i32,
        );
    }

    println!("Writing images to files");

    // let piece_path = format!("{}/piece{}.png", args.output, 0);
    // pieces[18].save(piece_path).unwrap();

    for (i, piece) in pieces.iter().enumerate() {
        let piece_path = format!("{}/piece{}.png", args.output, i);
        piece.save(piece_path).unwrap();
    }

    println!("Done!");

    Ok(())
}
