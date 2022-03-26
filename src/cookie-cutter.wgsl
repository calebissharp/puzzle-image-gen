@group(0) @binding(0) var img_buffer: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;

@stage(compute)
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
    // @builtin(workgroup_id) workgroup_id: vec3<u32>,
    // @builtin(local_invocation_id) local_id: vec3<u32>
) {

    var color  = textureLoad(
        img_buffer,
        vec2<i32>(
            i32(global_id.x),
            i32(global_id.y)
        ),
        // vec2<i32>(
        //     0,
        //     0
        // ),
        0
    );

    // color = 1. - color;
    color *= 0.5;
    // color[3] = 1.;

    // if (global_id.x % 4u != 0u && global_id.y % 4u != 0u) {
        textureStore(
            output_tex,
            vec2<i32>(i32(global_id.x), i32(global_id.y)),
            color
            // vec4<f32>(
            //     f32(global_id.x) / 1024.,
            //     f32(global_id.y) / 1024.,
            //     0.,
            //     f32(global_id.x) / 1024.
            // )
        );
    // }
}