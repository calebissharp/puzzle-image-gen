use nalgebra::Vector2;

pub struct Polygon {
  points: Vec<Vector2<f64>>,
}

impl Polygon {
  pub fn new(points: Vec<Vector2<f64>>) -> Polygon {
    Polygon { points }
  }

  pub fn check(&self, x: f64, y: f64) -> bool {
    let mut inside = false;
    let start = 0;
    let end = self.points.len() - 1;

    let len = (end - start);
    for i in 0..len {
      let j = if i == 0 { len - 1 } else { i - 1 };

      let xi = self.points[start + i][0];
      let yi = self.points[start + i][1];
      let xj = self.points[start + j][0];
      let yj = self.points[start + j][1];
      let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

      if intersect {
        inside = !inside;
      }
    }

    inside
  }
}
