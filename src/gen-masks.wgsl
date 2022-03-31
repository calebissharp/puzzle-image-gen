@stage(vertex)
fn vs_main(@builtin(vertex_index) in_vertex_index: u32, @location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    // let x = f32(i32(in_vertex_index) - 1);
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    let x = position[0];
    let y = position[1];
    return vec4<f32>(x, y, 0.0, 1.0);
}

@stage(fragment)
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1., 1., 1., 1.0);
}