use nalgebra::{Matrix4, Matrix4x2, RowVector2, RowVector4};
use std::error::Error;

#[derive(Debug)]
pub struct SampleError {
  details: String,
}

impl SampleError {
  fn new(msg: &str) -> SampleError {
    SampleError {
      details: msg.to_string(),
    }
  }
}

impl std::fmt::Display for SampleError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "{}", self.details)
  }
}

impl Error for SampleError {
  fn description(&self) -> &str {
    &self.details
  }
}

pub struct CatmullRomSpline {
  pub control_points: Vec<[f64; 2]>,
  tension: f64,
  basis: Matrix4<f64>,
  closed: bool,
}

impl CatmullRomSpline {
  pub fn new(mut control_points: Vec<[f64; 2]>, tension: f64, closed: bool) -> CatmullRomSpline {
    let basis = Matrix4::from_rows(&[
      RowVector4::new(0., 2., 0., 0.),
      RowVector4::new(-1., 0., 1., 0.),
      RowVector4::new(2., -5., 4., -1.),
      RowVector4::new(-1., 3., -3., 1.),
    ]);

    if closed {
      control_points.push(control_points[0]);
    }

    CatmullRomSpline {
      control_points,
      tension,
      basis,
      closed,
    }
  }

  /// Samples along the spline. x should be greater than 0 and less than the
  /// number of control points - 1
  pub fn sample(&self, x: f64) -> Result<RowVector2<f64>, SampleError> {
    if x < 0. {
      return Err(SampleError::new("x must be greater than 0"));
    } else if x > (self.control_points.len() - 1) as f64 {
      return Err(SampleError::new(
        "x must be less than the number of control points - 1",
      ));
    }

    let s = x.trunc() as usize;

    // Just return the actual control point if x has no decimal
    if x.fract() == 0. {
      return Ok(RowVector2::from_row_slice(&self.control_points[s]));
    }

    let point = self.control_points[s];
    let next_point = self.control_points[s + 1];
    let prev_point = if s == 0 {
      if self.closed {
        self.control_points[self.control_points.len() - 2]
      } else {
        point
      }
    } else {
      self.control_points[s - 1]
    };
    let next_next_point = if s >= self.control_points.len() - 2 {
      if self.closed {
        self.control_points[1]
      } else {
        next_point
      }
    } else {
      self.control_points[s + 2]
    };

    let control_vec = Matrix4x2::from_row_slice(&[
      prev_point[0],
      prev_point[1],
      point[0],
      point[1],
      next_point[0],
      next_point[1],
      next_next_point[0],
      next_next_point[1],
    ]);

    let u = x - s as f64;

    let coefficients = RowVector4::new(1., u, u.powi(2), u.powi(3));

    let p = self.tension * coefficients * self.basis * control_vec;

    Ok(p)
  }
}
