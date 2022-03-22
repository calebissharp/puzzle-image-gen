use clap::Parser;
use image::error::{DecodingError, ImageError};
use image::io::Reader as ImageReader;
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageBuffer, Pixel, RgbImage, Rgba, RgbaImage,
};
use imageproc::drawing::{
    draw_antialiased_line_segment_mut, draw_cross_mut, draw_line_segment_mut, Blend,
};
use imageproc::pixelops::interpolate;
use nalgebra::{DMatrix, DVector, Matrix4, RowVector4, Vector4};
use splines::iter::Iter;
use splines::{Interpolation, Key, Spline};
use std::env;
use std::f32;
use std::io::Cursor;
use std::path::Path;
use std::vec::Vec;

const MAX_JOINER_SIZE: u32 = 16;

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

fn draw_grid(img: &mut RgbaImage) -> Result<(), ImageError> {
    // img.put_pixel(x: u32, y: u32, pixel: P)
    // let control_points = vec![
    //     Key::new(0., [0., 0.], Interpolation::CatmullRom),
    //     Key::new(5., [5., 5.], Interpolation::CatmullRom),
    //     Key::new(10., [0., 0.], Interpolation::CatmullRom),
    // ];
    // let spline = Spline::from_vec(control_points);

    // for i in 0..10 {
    //     println!("{:?}", spline.sample(0.));
    // }

    let red = Rgba([255u8, 0, 0, 255u8]);

    // draw_line_segment_mut(img, (5., 5.), (500., 500.), red);
    draw_antialiased_line_segment_mut(img, (5, 5), (500, 500), red, |left, right, weight| {
        interpolate(left, right, 1.1)
    });

    Ok(())
}

fn get_curve_points(
    pts: &Vec<(u32, u32)>,
    tension: f32,
    is_closed: bool,
    num_of_segments: u32,
) -> Vec<(f32, f32)> {
    let mut res = Vec::<(f32, f32)>::new();

    let mut _pts = pts.clone();
    // The algorithm require a previous and next point to the actual point array.
    // Check if we will draw closed or open curve.
    // If closed, copy end points to beginning and first points to end
    // If open, duplicate first points to beginning, end points to end
    if is_closed {
        _pts.splice(0..0, vec![pts[pts.len() - 1]]);
        _pts.splice(0..0, vec![pts[pts.len() - 1]]);
        _pts.push(pts[0]);
    } else {
        _pts.splice(0..0, vec![pts[0]]);
        _pts.push(pts[pts.len() - 1]);
    }
    // ok, lets start..
    // 1. loop goes through point array
    // 2. loop goes through each segment between the 2 pts + 1e point before and after
    for i in 1..(_pts.len() - 2) {
        for t in 0..num_of_segments {
            let current_point = _pts[i];
            let prev_point = _pts[i - 1];
            let next_point = _pts[i + 1];
            let next_next_point = _pts[i + 2];

            let t1x = (next_point.0 as f32 - (prev_point.0 as f32)) * tension as f32;
            let t1y = (next_point.1 as f32 - (prev_point.1) as f32) * tension as f32;

            let t2x = (next_next_point.0 as f32 - (current_point.0 as f32)) * tension as f32;
            let t2y = (next_next_point.1 as f32 - (current_point.1) as f32) * tension as f32;

            let du_dx = 1. / (next_point.0 - current_point.0) as f32;
            let du_dy = 1. / (next_point.1 - current_point.1) as f32;

            let dp1_x = t1x * du_dx;
            let dp1_y = t1y * du_dy;

            let dp2_x = t2x * du_dx;
            let dp2_y = t2y * du_dy;

            // calc step
            let st = t as f32 / num_of_segments as f32;

            // calc cardinals
            let c1 = 2. * st.powi(3) - 3. * st.powi(2) + 1.;
            let c2 = -(2. * st.powi(3)) + 3. * st.powi(2);
            let c3 = st.powi(3) - 2. * st.powi(2) + st;
            let c4 = st.powi(3) - st.powi(2);

            // calc x and y cords with common control vectors
            let x = c1 * current_point.0 as f32 + c2 * next_point.0 as f32 + c3 * t1x + c4 * t2x;
            let y = c1 * current_point.1 as f32 + c2 * next_point.1 as f32 + c3 * t1y + c4 * t2y;

            //store points in array
            res.push((x.try_into().unwrap(), y.try_into().unwrap()));
        }
    }

    res
}

fn cardinal_spline(control_points: Vec<(f64, f64)>, t: f64, begin: (f64, f64), end: (f64, f64)) {
    let num_segments = control_points.len() - 1;
    let dim = (num_segments) * 4;
    let mut y_vec = vec![];
    let mut x_vec = vec![];

    let points_iter = control_points.windows(2).enumerate();

    let m = DMatrix::from_row_slice(
        dim,
        dim,
        &points_iter
            .clone()
            .flat_map(|(i, points)| {
                let mut rows: Vec<Vec<f64>> = Vec::with_capacity(4);
                let prev_point = if i > 0 { control_points[i - 1] } else { begin };
                let point = points[0];
                let next_point = points[1];
                let next_next_point = if i == control_points.len() - 2 {
                    end
                } else {
                    control_points[i + 2]
                };

                println!(
                    "i: {}, p_i-1: {:?}, p_i: {:?}, p_i+1: {:?}, p_i+2: {:?}",
                    i, prev_point, point, next_point, next_next_point
                );

                let dudx = 1. / (next_point.0 - point.0);
                let dudy = 1. / (next_point.1 - point.1);

                let t1x = (1. - t) * (next_point.0 - prev_point.0);
                let t1y = (1. - t) * (next_point.1 - prev_point.1);

                let t2x = (1. - t) * (next_next_point.0 - point.0);
                let t2y = (1. - t) * (next_next_point.1 - point.1);

                let dp1x = t1x * dudx;
                let dp1y = t1y * dudy;

                let dp2x = t2x * dudx;
                let dp2y = t2y * dudy;

                // Curve through p_i at u_i = 0
                let mut row1 = vec![0.; dim];
                row1[4 * i] = 1.;
                rows.push(row1);
                y_vec.push(point.1);
                x_vec.push(point.0);

                // Curve through p_i+1 at u_i = 1
                let mut row2 = vec![0.; dim];
                row2[4 * i] = 1.;
                row2[4 * i + 1] = 1.;
                row2[4 * i + 2] = 1.;
                row2[4 * i + 3] = 1.;
                rows.push(row2);
                y_vec.push(next_point.1);
                x_vec.push(next_point.0);

                // Slope at p_i
                let mut row3 = vec![0.; dim];
                row3[4 * i + 1] = 1.;
                rows.push(row3);
                y_vec.push(dp1y);
                x_vec.push(dp1x);

                // Slope at p_i+1
                let mut row4 = vec![0.; dim];
                row4[4 * i + 1] = 1.;
                row4[4 * i + 2] = 2.;
                row4[4 * i + 3] = 3.;
                rows.push(row4);
                y_vec.push(dp2y);
                x_vec.push(dp2x);
                println!("{:?}", x_vec);
                println!("{:?}", y_vec);

                rows
            })
            .flatten()
            .collect::<Vec<f64>>(),
    );

    x_vec.append(&mut y_vec);
    let xy = DMatrix::from_vec(dim, 2, x_vec);

    // println!("{}", m);
    // println!("{}", xy);

    let s = m.try_inverse().unwrap() * xy;
    println!("{}", s);
}

fn cubic_spline(control_points: Vec<(f64, f64)>, s0: f64, sn: f64) {
    let num_segments = control_points.len() - 1;
    let dim = (num_segments) * 4;
    // let mut m_x = DMatrix::<f64>::zeros(dim, dim);
    // let mut m_y = DMatrix::<f64>::zeros(dim, dim);

    let mut y_vec = vec![];
    let mut x_vec = vec![];

    let mut _control_points = control_points.clone();
    _control_points.push(control_points[control_points.len() - 1]);

    let points_iter = _control_points.windows(3).enumerate();

    let m_x = DMatrix::from_row_slice(
        dim,
        dim,
        &points_iter
            .clone()
            .flat_map(|(i, points)| {
                let mut rows: Vec<Vec<f64>> = vec![];
                let point = points[0];
                let next_point = points[1];
                let next_next_point = points[2];
                // First segment
                if i == 0 {
                    // Slope at 0
                    let b0 = (next_point.0 - point.0) * s0;
                    let mut row1 = vec![0.; dim];
                    row1[1] = 1.;
                    rows.push(row1);
                    y_vec.push(b0);
                    x_vec.push(b0);
                }

                // Curve through (x_i, y_i): a_i = y_i
                let mut row1 = vec![0.; dim];
                row1[4 * i] = 1.;
                rows.push(row1);
                y_vec.push(point.1);
                x_vec.push(point.0);

                // Curve through (x_i+1, y_i+1)
                let mut row2 = vec![0.; dim];
                row2[4 * i] = 1.;
                row2[4 * i + 1] = 1.;
                row2[4 * i + 2] = 1.;
                row2[4 * i + 3] = 1.;
                rows.push(row2);
                y_vec.push(next_point.1);
                x_vec.push(next_point.0);

                if i == num_segments - 1 {
                    // Slope at n
                    let mut row3 = vec![0.; dim];
                    row3[4 * i + 1] = 1.; // b_i
                    row3[4 * i + 2] = 2.; // c_i
                    row3[4 * i + 3] = 3.; // d_i
                    rows.push(row3);
                    y_vec.push((point.0 - _control_points[i - 1].0) * sn);
                    x_vec.push((point.1 - _control_points[i - 1].1) * sn);
                } else {
                    // Slopes match
                    let mut row3 = vec![0.; dim];
                    row3[4 * i + 1] = 1.; // b_i
                    row3[4 * i + 2] = 2.; // c_i
                    row3[4 * i + 3] = 3.; // d_i
                    rows.push(row3);
                    y_vec.push((next_point.0 - point.0) / (next_next_point.0 - next_point.0));
                    y_vec.push((next_point.1 - point.1) / (next_next_point.1 - next_point.1));

                    // Curvatures match
                    let mut row4 = vec![0.; dim];
                    row4[4 * i + 2] = 2.; // c_i
                    row4[4 * i + 3] = 6.; // d_i
                    rows.push(row4);
                    y_vec.push(
                        (next_point.0 - point.0).powi(2)
                            / (next_next_point.0 - next_point.0).powi(2)
                            * 2.,
                    );
                    x_vec.push(
                        (next_point.1 - point.1).powi(2)
                            / (next_next_point.1 - next_point.1).powi(2)
                            * 2.,
                    );
                }

                rows
            })
            .flatten()
            .collect::<Vec<f64>>(),
    );

    let flipped_points: Vec<(f64, f64)> = _control_points
        .clone()
        .iter()
        .map(|point| (point.1, point.0))
        .collect();

    // let flipped_points_iter = flipped_points.windows(3).enumerate();

    // let mut x_vec = vec![];
    // let m_y = DMatrix::from_row_slice(
    //     dim,
    //     dim,
    //     &flipped_points_iter
    //         .flat_map(|(i, points)| {
    //             let mut rows: Vec<Vec<f64>> = vec![];
    //             let point = points[0];
    //             let next_point = points[1];
    //             let next_next_point = points[2];
    //             // First segment
    //             if i == 0 {
    //                 // Slope at 0
    //                 let b0 = (next_point.0 - point.0) * s0;
    //                 let mut row1 = vec![0.; dim];
    //                 row1[1] = 1.;
    //                 rows.push(row1);
    //                 x_vec.push(b0)
    //             }

    //             // Curve through (x_i, y_i): a_i = y_i
    //             let mut row1 = vec![0.; dim];
    //             row1[4 * i] = 1.;
    //             rows.push(row1);
    //             x_vec.push(point.1);

    //             // Curve through (x_i+1, y_i+1)
    //             let mut row2 = vec![0.; dim];
    //             row2[4 * i] = 1.;
    //             row2[4 * i + 1] = 1.;
    //             row2[4 * i + 2] = 1.;
    //             row2[4 * i + 3] = 1.;
    //             rows.push(row2);
    //             x_vec.push(next_point.1);

    //             if i == num_segments - 1 {
    //                 // Slope at n
    //                 let mut row3 = vec![0.; dim];
    //                 row3[4 * i + 1] = 1.; // b_i
    //                 row3[4 * i + 2] = 2.; // c_i
    //                 row3[4 * i + 3] = 3.; // d_i
    //                 rows.push(row3);
    //                 x_vec.push((point.0 - _control_points[i - 1].0) * sn);
    //             } else {
    //                 // Slopes match
    //                 let mut row3 = vec![0.; dim];
    //                 row3[4 * i + 1] = 1.; // b_i
    //                 row3[4 * i + 2] = 2.; // c_i
    //                 row3[4 * i + 3] = 3.; // d_i
    //                 row3[4 * i + 5] =
    //                     -(next_point.0 - point.0) / (next_next_point.0 - next_point.0); // b_i+1
    //                 rows.push(row3);
    //                 x_vec.push(0.);

    //                 // Curvatures match
    //                 let mut row4 = vec![0.; dim];
    //                 row4[4 * i + 2] = 2.; // c_i
    //                 row4[4 * i + 3] = 6.; // d_i
    //                 row4[4 * i + 6] = -(next_point.0 - point.0).powi(2)
    //                     / (next_next_point.0 - next_point.0).powi(2)
    //                     * 2.; // c_i+1
    //                 rows.push(row4);
    //                 x_vec.push(0.);
    //             }

    //             rows
    //         })
    //         .flatten()
    //         .collect::<Vec<f64>>(),
    // );

    println!("{}", m_x);

    y_vec.append(&mut x_vec);

    let y = DMatrix::from_vec(dim, 2, y_vec);
    let sx = m_x.try_inverse().unwrap() * y;
    // let x = DVector::from_vec(x_vec);
    // let sy = m_y.try_inverse().unwrap() * x;

    println!("{}", sx);
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    cardinal_spline(
        vec![(1., 1.), (5., 20.), (10., 5.)],
        0.5,
        (-5., 0.),
        (15., 0.),
    );

    // let control_points = vec![(0., 1.), (2., 2.), (5., 0.), (8., 0.)];

    // cubic_spline(control_points, 2., 2.);

    return Ok(());

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

    // draw_grid(&mut img).unwrap();

    let padding = MAX_JOINER_SIZE;

    let pieces = slice_image(&img, 16, 16, padding).unwrap();

    let white = Rgba([255, 255, 255, 255]);
    let red = Rgba([255, 0, 0, 255]);
    let green = Rgba([0, 255, 0, 255]);
    let blue = Rgba([0, 0, 255, 255]);

    let control_points = vec![
        Key::new(0., [0., 0.], Interpolation::CatmullRom),
        Key::new(5., [5., 5.], Interpolation::CatmullRom),
        Key::new(10., [0., 0.], Interpolation::CatmullRom),
    ];
    let spline = Spline::from_vec(control_points);

    // for i in 0..10 {
    //     println!("{:?}", spline.sample(0.));
    // }

    let control_points = vec![
        (0, 25),
        (50, 25),
        (40, 10),
        (75, 0),
        (110, 10),
        (150, 25),
        (200, 25),
    ];

    let curve_points = get_curve_points(&control_points, 0.5, false, 8);

    let mut curve_points_iter = curve_points.iter().peekable();
    while let Some(curve_point) = curve_points_iter.next() {
        // draw_cross_mut(
        //     &mut img,
        //     green,
        //     curve_point[0].round() as i32,
        //     curve_point[1].round() as i32,
        // );

        if let Some(next_point) = curve_points_iter.peek() {
            println!("{:?} -> {:?}", curve_point, next_point);
            draw_antialiased_line_segment_mut(
                &mut img,
                (curve_point.0.round() as i32, curve_point.1.round() as i32),
                (next_point.0.round() as i32, next_point.1.round() as i32),
                white,
                interpolate,
            )
        }
    }

    img.save("test-image.png").unwrap();

    println!("Generated {} pieces", pieces.len());

    println!("Writing images to files");

    for (i, piece) in pieces.iter().enumerate() {
        let piece_path = format!("{}/piece{}.png", args.output, i);
        piece.save(piece_path).unwrap();
    }

    Ok(())
}
