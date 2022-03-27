pub fn find_closest_multiple(n: u32, x: u32) -> u32 {
    ((n - 1) | (x - 1)) + 1
}
