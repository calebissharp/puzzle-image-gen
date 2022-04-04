use rand::Rng;
use std::f64::consts::PI;

use crate::{spline::CatmullRomSpline, util::find_closest_multiple};

pub enum Side {
    LEFT,
    BOTTOM,
    RIGHT,
}

fn rotate_point(point: &mut [f64; 2], orig_x: f64, orig_y: f64, angle: f64) {
    let s = angle.sin();
    let c = angle.cos();

    point[0] -= orig_x;
    point[1] -= orig_y;

    let x = point[0] * c - point[1] * s;
    let y = point[0] * s + point[1] * c;

    point[0] = x + orig_x;
    point[1] = y + orig_y;
}

fn rotate_points(points: &mut Vec<[f64; 2]>, x: f64, y: f64, angle: f64) {
    for point in points.iter_mut() {
        rotate_point(point, x, y, angle);
    }
}

fn reflect_point_vertical(point: &mut [f64; 2], y: f64) {
    point[1] -= y;
    point[1] *= -1.;
    point[1] += y;
}

fn reflect_points_vertical(points: &mut Vec<[f64; 2]>, y: f64) {
    for point in points.iter_mut() {
        reflect_point_vertical(point, y);
    }
}

pub struct Edge {
    pub points: Vec<f32>,
    side_length: u32,
    padding: u32,
}

impl Edge {
    pub fn gen_edge(side_length: u32, padding: u32) -> Edge {
        let mut rng = rand::thread_rng();

        let joiner = padding as f64;

        let width = side_length as f64;

        let half = width / 2.;

        let inverted = rng.gen::<bool>();

        let mut control_points = vec![
            [joiner, joiner],                                    // left
            [joiner + half - width / 3., joiner + joiner / 1.5], // left 2
            [joiner + half - width / 6., joiner + joiner / 2.],  // left 3
            [joiner + half - width / 5., joiner / 1.5],          // left 4
            [joiner + half + rng.gen::<f64>(), joiner / 5.],     // middle
            [joiner + half + width / 5., joiner / 1.5],          // right 1
            [joiner + half + width / 6., joiner + joiner / 2.],  // right 2
            [joiner + half + width / 3., joiner + joiner / 1.5], // right 3
            [joiner + width, joiner],
        ];

        if inverted {
            reflect_points_vertical(&mut control_points, joiner);
        }

        let spline = CatmullRomSpline::new(control_points, 0.5, false);

        let step = 5;

        let mut curve_points = (0..(spline.control_points.len() - 1) * step)
            .map(|x| spline.sample(x as f64 / step as f64).unwrap().transpose())
            .flat_map(|points| [points[0] as f32, points[1] as f32])
            .collect::<Vec<f32>>();

        curve_points.pop();
        curve_points.pop();

        Edge {
            points: curve_points,
            side_length,
            padding
            // points: points
            //     .iter()
            //     .flatten()
            //     .map(|x| x.to_owned())
            //     .collect::<Vec<f64>>(),
        }
    }

    fn points_side(&self, side: Side) -> Vec<f32> {
        let mut points = self
            .points
            .clone()
            .chunks(2)
            .map(|point| [point[0] as f64, point[1] as f64])
            .collect::<Vec<[f64; 2]>>();

        let half = self.side_length as f64 / 2.;

        let angle = match side {
            Side::RIGHT => PI / 2.,
            Side::BOTTOM => PI,
            Side::LEFT => PI * 1.5,
        };

        let invert = match side {
            Side::RIGHT => false,
            Side::BOTTOM => true,
            Side::LEFT => true,
        };

        if invert {
            reflect_points_vertical(&mut points, self.padding as f64);
        }

        rotate_points(
            &mut points,
            half + self.padding as f64,
            half + self.padding as f64,
            angle,
        );

        points
            .iter()
            .flatten()
            .map(|x| x.to_owned() as f32)
            .collect::<Vec<f32>>()
    }

    pub fn flat(side_length: u32, padding: u32) -> Edge {
        let joiner = padding as f32;

        let width = side_length as f32;

        let mut points = vec![joiner, joiner, joiner + width, joiner];

        points.pop();
        points.pop();

        Edge {
            points,
            side_length,
            padding,
        }
    }
}

#[derive(Debug)]
pub struct PuzzleDimensions {
    pub pieces_x: u32,
    pub pieces_y: u32,
    pub num_pieces: u32,
    pub piece_padding: u32,
    pub piece_width: u32,
    pub piece_height: u32,
    pub padded_piece_width: u32,
    pub width: u32,
    pub height: u32,
    pub padded_width: u32,
}

impl PuzzleDimensions {
    pub fn new(width: u32, height: u32, pieces_x: u32, pieces_y: u32) -> PuzzleDimensions {
        let piece_padding = (width / pieces_x) / 6;

        let padded_width = find_closest_multiple(width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let padded_piece_width = padded_width / pieces_x;

        PuzzleDimensions {
            width,
            height,
            padded_width,
            pieces_x,
            pieces_y,
            num_pieces: pieces_x * pieces_y,
            piece_width: width / pieces_x + piece_padding * 2,
            piece_height: height / pieces_y + piece_padding * 2,
            padded_piece_width,
            piece_padding,
        }
    }
}

pub struct Puzzle {
    pub dimensions: PuzzleDimensions,
    edges: Vec<Edge>,
}

impl Puzzle {
    pub fn new(width: u32, height: u32, pieces_x: u32, pieces_y: u32) -> Puzzle {
        let dimensions = PuzzleDimensions::new(width, height, pieces_x, pieces_y);

        let total_edges = pieces_x * pieces_y * 2 // Bottom and right edges of each
        +  pieces_y // Left edge
        +  pieces_x; // Top edge

        let mut edges = Vec::with_capacity(total_edges as usize);

        let mut top_edges = (0..pieces_x)
            .map(|_| Edge::flat(dimensions.piece_width, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        let mut bottom_edges = (0..pieces_x)
            .map(|_| Edge::flat(dimensions.piece_width, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        let mut interior_horiz = (0..pieces_x * (pieces_y - 1))
            .map(|_| Edge::gen_edge(dimensions.piece_width, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        let mut interior_vert = (0..(pieces_x - 1) * pieces_y)
            .map(|_| Edge::gen_edge(dimensions.piece_height, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        let mut left_edges = (0..pieces_y)
            .map(|_| Edge::flat(dimensions.piece_height, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        let mut right_edges = (0..pieces_y)
            .map(|_| Edge::flat(dimensions.piece_height, dimensions.piece_padding))
            .collect::<Vec<Edge>>();

        edges.append(&mut top_edges);
        edges.append(&mut interior_horiz);
        edges.append(&mut bottom_edges);

        edges.append(&mut left_edges);
        edges.append(&mut interior_vert);
        edges.append(&mut right_edges);

        Puzzle { dimensions, edges }
    }

    pub fn top_edge(&self, x: u32, y: u32) -> &Edge {
        &self.edges[(y * self.dimensions.pieces_x + x) as usize]
    }

    pub fn bottom_edge(&self, x: u32, y: u32) -> &Edge {
        &self.edges[(y * self.dimensions.pieces_x + x + self.dimensions.pieces_x) as usize]
    }

    pub fn left_edge(&self, x: u32, y: u32) -> &Edge {
        &self.edges[(self.dimensions.num_pieces
            + self.dimensions.pieces_x
            + self.dimensions.pieces_y * x
            + y) as usize]
    }

    pub fn right_edge(&self, x: u32, y: u32) -> &Edge {
        &self.edges[(self.dimensions.num_pieces
            + self.dimensions.pieces_x
            + self.dimensions.pieces_y * x
            + y
            + self.dimensions.pieces_y) as usize]
    }

    pub fn get_piece_points(&self, x: u32, y: u32) -> Vec<f32> {
        let mut top_edge = self.top_edge(x, y).points.clone();
        let bottom_edge = &self.bottom_edge(x, y).points_side(Side::BOTTOM);

        let left_edge = &self.left_edge(x, y).points_side(Side::LEFT);
        let right_edge = &self.right_edge(x, y).points_side(Side::RIGHT);

        top_edge.append(&mut right_edge.clone());
        top_edge.append(&mut bottom_edge.clone());
        top_edge.append(&mut left_edge.clone());

        top_edge
    }
}
