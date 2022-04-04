struct Locals {
    position: vec2<f32>,
    tex_coords: vec2<f32>
};

struct Globals {
    projection: mat4x4<f32>
};

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>
};

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @builtin(position) position: vec4<f32>
};

@group(0) @binding(0) var img_buffer: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<storage, read> globals: Globals;

@group(1) @binding(0) var<uniform> r_locals: Locals;

@stage(vertex)
fn vs_main(@builtin(vertex_index) in_vertex_index: u32, in: VertexInput) -> VertexOutput {
    // let x = f32(i32(in_vertex_index) - 1);
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    let actual_pos = in.position;
    // let actual_pos = position;
    let x = actual_pos[0] + r_locals.position[0];
    let y = actual_pos[1] + r_locals.position[1];

    var out: VertexOutput;
    out.tex_coord = in.tex_coords;
    out.position = globals.projection * vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(img_buffer));
    let tex = textureSample(
        img_buffer,
        s,
        // vec2<f32>(position[0] / dimensions[0], position[1] / dimensions[1])
        vec2<f32>(
            (in.tex_coord[0] / dimensions[0]),
            (in.tex_coord[1] / dimensions[1])
        )
    );
    return tex;
    // return vec4<f32>(1., 1., 1., 1.0);
}