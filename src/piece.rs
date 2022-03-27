use std::f64::consts::PI;

pub const MAX_JOINER_SIZE: u32 = 16;

pub enum Side {
    TOP,
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

fn reflect_point_horizontal(point: &mut [f64; 2], orig_x: f64) {
    point[0] -= orig_x;
    point[0] *= -1.;
    point[0] += orig_x;
}

fn reflect_points_horizontal(points: &mut Vec<[f64; 2]>, x: f64) {
    for point in points.iter_mut() {
        reflect_point_horizontal(point, x);
    }
}

pub struct Edge {
    pub points: Vec<[f64; 2]>,
}

impl Edge {
    pub fn gen_edge(side: Side, side_length: u32, padding: u32) -> Edge {
        let joiner = padding as f64;

        let width = side_length as f64;

        let half = width / 2.;

        let mut points = vec![
            [joiner, joiner],
            [joiner + half - width / 3., joiner + joiner / 1.5],
            [joiner + half - width / 6., joiner + joiner / 2.],
            [joiner + half - width / 5., joiner / 1.5],
            [joiner + half, joiner / 5.],
            [joiner + half + width / 5., joiner / 1.5],
            [joiner + half + width / 6., joiner + joiner / 2.],
            [joiner + half + width / 3., joiner + joiner / 1.5],
        ];

        if matches!(side, Side::RIGHT) || matches!(side, Side::BOTTOM) {
            reflect_points_vertical(&mut points, joiner);
        }

        let angle = match side {
            Side::TOP => 0.,
            Side::RIGHT => PI / 2.,
            Side::BOTTOM => PI,
            Side::LEFT => PI * 1.5,
        };

        rotate_points(&mut points, half + joiner, half + joiner, angle);

        Edge { points }
    }
}
