@group(0) @binding(0) var img_buffer: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var output_tex: texture_storage_3d<rgba8unorm, write>;
// @group(0) @binding(3) var<storage, read> points_buffer: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> points_buffer: array<vec2<f32>>;

let PIECE_WIDTH: u32 = $PIECE_WIDTH$u;
let PIECE_HEIGHT: u32 = $PIECE_HEIGHT$u;
let NUM_PIECES_X: u32 = $NUM_PIECES_X$u;
let NUM_PIECES_Y: u32 = $NUM_PIECES_Y$u;

// Check if point (x, y) is inside points_buffer polygon
fn check_inside(x: f32, y: f32) -> bool {
    var inside: bool = false;
    var start = 0u;

    var end = arrayLength(&points_buffer);
    var len = (end - start);

    var i: i32 = 0;
    loop {
        if !(u32(i) <= len) { 
            break;
        }

        var a = u32(i);
        // Select the previous point unless we're at the first point, then
        // select the last point
        var j = select(a - 1u, len - a, a == 0u);

        var xi = points_buffer[start + a][0];
        var yi = points_buffer[start + a][1];
        var xj = points_buffer[start + j][0];
        var yj = points_buffer[start + j][1];
        var intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

        if (intersect) {
            inside = !inside;
        }

        continuing {
            i = i + 1;
        }
    }

    return inside;
}

@stage(compute)
// @workgroup_size($PIECE_WIDTH$, $PIECE_HEIGHT$)
// @workgroup_size(16, 16, 1)
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
    // @builtin(workgroup_id) workgroup_id: vec3<u32>,
    // @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // if  (
    //     global_id.x > (global_id.z + 1u) * $PIECE_WIDTH$u ||
    //     global_id.x < global_id.z * $PIECE_WIDTH$u
    // ) {
    //     return;
    // }
    // Only store texels that are contained within the mask polygon
    if check_inside(f32(global_id.x), f32(global_id.y)) {
        var color = textureLoad(
            img_buffer,
            vec2<i32>(
                i32(global_id.x),
                i32(global_id.y)
            ),
            0
        );

        textureStore(
            output_tex,
            vec3<i32>(i32(global_id.x), i32(global_id.y), i32(global_id.z)),
            color
        );
    }
}
