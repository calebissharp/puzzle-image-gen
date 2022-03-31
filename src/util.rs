pub fn find_closest_multiple(n: u32, x: u32) -> u32 {
    ((n - 1) | (x - 1)) + 1
}

pub fn normalize<T: num::Num + Copy>(value: T, min: T, max: T) -> T {
    (value - min) / (max - min)
}

pub fn normalize_range<T: num::Num + Copy>(
    value: T,
    old_min: T,
    old_max: T,
    new_min: T,
    new_max: T,
) -> T {
    let old_range = old_max - old_min;
    let new_range = new_max - new_min;
    (((value - old_min) * new_range) / old_range) + new_min
}
