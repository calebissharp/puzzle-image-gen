@group(0) @binding(0) var img_buffer: texture_2d<u32>;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(outputTex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4<f32>(1.,1.,1.,1.));
}