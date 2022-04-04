mod piece;
mod spline;
mod util;

use bytemuck::{Pod, Zeroable};
use image::{open, RgbaImage};
use image::{EncodableLayout, GenericImage};
use log::debug;
use maligned::{align_first, aligned, Aligned, A256};
use nalgebra::{
    Matrix4, Orthographic3, Point3, Quaternion, Similarity2, Similarity3, Transform3,
    UnitQuaternion, Vector2, Vector3, Vector4,
};
use piece::Puzzle;
use rayon::prelude::*;
use spline::CatmullRomSpline;
use std::borrow::Cow;
use std::f32;
use std::mem;
use std::time::Instant;
use util::find_closest_multiple;
use wgpu::util::DeviceExt;

#[repr(C, align(256))]
#[derive(Clone, Copy, Zeroable)]
struct Locals {
    position: [f32; 2],
    tex_coords: [f32; 2],
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Globals {
    projection: [f32; 16], // 4x4 matrix
}

async fn run() {
    execute_gpu().await;
}

async fn execute_gpu() -> Option<()> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let img = open("./test-images/chungus.jpg").unwrap().to_rgba8();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::POLYGON_MODE_LINE,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    // execute_gpu_inner(&device, &queue, img).await;
    draw_masks(&device, &queue, img).await;

    Some(())
}

async fn draw_masks(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut src_img: RgbaImage,
) -> Option<()> {
    // let puzzle = Puzzle::new(3897, 3801, 16, 16);
    let puzzle = Puzzle::new(src_img.width(), src_img.height(), 32, 32);

    let mut img = RgbaImage::new(puzzle.dimensions.padded_width, puzzle.dimensions.height);
    println!(
        "Padding image for GPU, Original: ({}x{}), New: ({}x{})",
        src_img.width(),
        src_img.height(),
        img.width(),
        img.height(),
    );
    img.copy_from(&src_img, 0, 0).unwrap();

    let render_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("./gen-masks.wgsl"))),
    });

    let generate_start = Instant::now();

    let mut buffer_sizes = vec![];
    let mut buffer_offsets = vec![];
    let mut all_tris = vec![];
    let mut locals_data = Vec::<Locals>::with_capacity(puzzle.dimensions.num_pieces as usize);

    for x in 0..puzzle.dimensions.pieces_x {
        for y in 0..puzzle.dimensions.pieces_y {
            let points = puzzle.get_piece_points(x, y);

            let triangles = earcutr::earcut(&points, &vec![], 2);

            let mut tris = vec![];

            for point in triangles {
                // vertex coord
                tris.push(points[point * 2]);
                tris.push(points[point * 2 + 1]);

                // texture coord
                tris.push(
                    util::normalize_range(
                        points[point * 2],
                        0.,
                        puzzle.dimensions.piece_width as f32,
                        0.,
                        puzzle.dimensions.width as f32 / puzzle.dimensions.pieces_x as f32,
                    ) + (puzzle.dimensions.width as f32 / puzzle.dimensions.pieces_x as f32)
                        * x as f32
                        - puzzle.dimensions.piece_padding as f32,
                );
                tris.push(
                    util::normalize_range(
                        points[point * 2 + 1],
                        0.,
                        puzzle.dimensions.piece_height as f32,
                        0.,
                        puzzle.dimensions.height as f32 / puzzle.dimensions.pieces_y as f32,
                    ) + (puzzle.dimensions.height as f32 / puzzle.dimensions.pieces_y as f32)
                        * y as f32
                        - puzzle.dimensions.piece_padding as f32,
                );
                // tris.push(util::normalize_range(
                //     points[point * 2],
                //     0.,
                //     (puzzle.dimensions.piece_width + puzzle.dimensions.piece_padding * 2) as f32,
                //     -1.,
                //     1.
                //     // puzzle.dimensions.piece_width as f32
                //     //     + puzzle.dimensions.piece_padding as f32 * 2.,
                //     // -1. + (2. / puzzle.dimensions.pieces_x as f32) * x as f32,
                //     // -1. + (2. / puzzle.dimensions.pieces_x as f32) * (x as f32 + 1.),
                // ));
                // tris.push(util::normalize_range(
                //     points[point * 2 + 1],
                //     0.,
                //     (puzzle.dimensions.piece_height + puzzle.dimensions.piece_padding * 2) as f32,
                //     -1.,
                //     1.
                //     // puzzle.dimensions.piece_height as f32
                //     //     + puzzle.dimensions.piece_padding as f32 * 2.,
                //     // -1. + (2. / puzzle.dimensions.pieces_y as f32) * y as f32,
                //     // -1. + (2. / puzzle.dimensions.pieces_y as f32) * (y as f32 + 1.),
                // ));
            }

            buffer_sizes.push(tris.len());
            buffer_offsets.push(all_tris.len());
            locals_data.push(Locals {
                position: [
                    (x * puzzle.dimensions.piece_width + puzzle.dimensions.piece_padding * x * 2)
                        as f32,
                    (y * puzzle.dimensions.piece_height + puzzle.dimensions.piece_padding * y * 2)
                        as f32,
                ],
                tex_coords: [
                    (x * puzzle.dimensions.piece_width - puzzle.dimensions.piece_padding * 2 * x)
                        as f32,
                    (y * puzzle.dimensions.piece_height - puzzle.dimensions.piece_padding * 2 * y)
                        as f32,
                ],
                _pad: 0,
            });

            all_tris.append(&mut tris);
        }
    }

    println!(
        "Generated {} tris in {}ms",
        all_tris.len() / 3,
        generate_start.elapsed().as_millis()
    );

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        contents: &bytemuck::cast_slice(&all_tris),
        label: None,
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::UNIFORM,
    });

    let mask_output_size = wgpu::Extent3d {
        width: util::find_closest_multiple(
            (puzzle.dimensions.padded_piece_width + puzzle.dimensions.piece_padding * 2)
                * puzzle.dimensions.pieces_x,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ),
        height: (puzzle.dimensions.piece_height + puzzle.dimensions.piece_padding * 2)
            * puzzle.dimensions.pieces_y,
        depth_or_array_layers: 1,
    };

    let puzzle_texture_extent = wgpu::Extent3d {
        width: puzzle.dimensions.width,
        height: puzzle.dimensions.height,
        depth_or_array_layers: 1,
    };
    let puzzle_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        mip_level_count: 1,
        sample_count: 1,
        size: puzzle_texture_extent,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });
    let puzzle_texture_view = puzzle_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let puzzle_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let sample_count = 1;

    let mask_output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mask output texture"),
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        mip_level_count: 1,
        sample_count,
        size: mask_output_size,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
    });

    let mask_output_texture_view =
        mask_output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mask_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: (mask_output_size.width * mask_output_size.height * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
    let padded_locals_data = unsafe {
        std::slice::from_raw_parts(
            locals_data.as_ptr() as *const u8,
            locals_data.len() * uniform_alignment as usize,
        )
    };

    // Buffer containg info like piece position and texture coordinates
    let locals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: (puzzle.dimensions.num_pieces * uniform_alignment) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&locals_buffer, 0, padded_locals_data);

    let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<Globals>() as u64,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let proj = Orthographic3::<f32>::new(
        0.,
        (puzzle.dimensions.pieces_x * puzzle.dimensions.piece_width
            + puzzle.dimensions.pieces_x * 2 * puzzle.dimensions.piece_padding) as f32,
        (puzzle.dimensions.pieces_y * puzzle.dimensions.piece_height
            + puzzle.dimensions.pieces_y * 2 * puzzle.dimensions.piece_padding) as f32,
        0.,
        10., // don't need z flipped, so put in opposite order
        0.,
    );

    queue.write_buffer(
        &globals_buffer,
        0,
        bytemuck::cast_slice(proj.as_matrix().as_slice()),
    );

    // Bind group layout with entries shared by all pieces
    let global_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Bind group layout with entries specific to each piece
    let local_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let render_mask_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render masks pipeline layout"),
            bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
            push_constant_ranges: &[],
        });

    let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &global_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&puzzle_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&puzzle_texture_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: globals_buffer.as_entire_binding(),
            },
        ],
    });

    let local_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &local_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &locals_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
            }),
        }],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Mask render pipeline"),
        layout: Some(&render_mask_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 16,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 4 * 2,
                        shader_location: 1,
                    },
                ],
                step_mode: wgpu::VertexStepMode::Vertex,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::default(),
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview: None,
    });

    queue.write_texture(
        puzzle_texture.as_image_copy(),
        &img,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * puzzle.dimensions.padded_width),
            rows_per_image: std::num::NonZeroU32::new(puzzle.dimensions.height),
        },
        puzzle_texture_extent,
    );

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &mask_output_texture_view,
                // resolve_target: Some(&output_texture_view),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&render_pipeline);
        rpass.set_bind_group(0, &global_bind_group, &[]);
        // let points_per_piece = vertices.len() as u32 / puzzle_dimensions.num_pieces;
        // for i in 0..puzzle.dimensions.num_pieces {
        // let offset = (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
        // rpass.set_bind_group(0, &mask_bind_group, &[]);
        for i in 0..(puzzle.dimensions.num_pieces as usize) {
            // let i = 31;
            let offset = (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
            // rpass.set_bind_group(1, &local_bind_group, &[offset]);
            rpass.set_bind_group(1, &local_bind_group, &[offset]);
            let buffer_size = buffer_sizes[i] as u64;
            // let slice = buffer.slice(..);
            let slice = buffer.slice(
                (buffer_offsets[i] as u64 * 4)
                    ..(buffer_offsets[i] as u64 * 4 + buffer_size * 4 as u64),
            );

            rpass.set_vertex_buffer(0, slice);
            rpass.draw(
                0..(buffer_size / 4) as u32,
                // 0..100,
                0..1,
            );
        }
        // }
    }

    encoder.copy_texture_to_buffer(
        mask_output_texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &mask_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(mask_output_size.width * 4),
                rows_per_image: std::num::NonZeroU32::new(mask_output_size.height),
            },
        },
        mask_output_size,
    );

    let render_start = Instant::now();

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    let render_slice = mask_staging_buffer.slice(..);
    let render_buffer_future = render_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    println!(
        "Took {}ms to render {} pieces",
        render_start.elapsed().as_millis(),
        puzzle.dimensions.num_pieces
    );

    if let Ok(()) = render_buffer_future.await {
        let data = render_slice.get_mapped_range();
        let result: &[u8] = bytemuck::cast_slice(&data);

        let saving_start = Instant::now();
        println!("Saving images...");
        let mut image = image::RgbaImage::from_raw(
            mask_output_size.width,
            mask_output_size.height,
            result.to_vec(),
        )
        .unwrap();

        let path = format!("test-image.png",);
        image.save(path).unwrap();

        println!(
            "Took {}ms to save images",
            saving_start.elapsed().as_millis()
        );
    } else {
        panic!("AHHH")
    }

    Some(())
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_img: RgbaImage,
) -> Option<()> {
    let start = std::time::Instant::now();

    // Image width needs to be a multiple of 256 in order
    // for copy_texture_to_buffer to work
    let mut image = RgbaImage::new(
        find_closest_multiple(src_img.width(), 256),
        src_img.height(),
    );
    println!(
        "Padding image for GPU, Original: ({}x{}), New: ({}x{})",
        src_img.width(),
        src_img.height(),
        image.width(),
        image.height(),
    );
    image.copy_from(&src_img, 0, 0).unwrap();

    let puzzle_dimensions = piece::PuzzleDimensions::new(src_img.width(), src_img.height(), 16, 16);

    let output_buffer_row_size = find_closest_multiple(puzzle_dimensions.piece_width * 4, 256);

    debug!("Puzzle dimensions: {:?}", puzzle_dimensions);

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            &include_str!("cookie-cutter.wgsl")
                .replace("$PIECE_WIDTH$", &puzzle_dimensions.piece_width.to_string())
                .replace(
                    "$PIECE_HEIGHT$",
                    &puzzle_dimensions.piece_height.to_string(),
                )
                .replace("$NUM_PIECES_X$", &puzzle_dimensions.pieces_x.to_string())
                .replace("$NUM_PIECES_Y$", &puzzle_dimensions.pieces_y.to_string())
                .replace("$NUM_PIECES$", &puzzle_dimensions.num_pieces.to_string()),
        )),
    });

    let render_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("./gen-masks.wgsl"))),
    });

    let texture_extent = wgpu::Extent3d {
        width: puzzle_dimensions.padded_width,
        height: puzzle_dimensions.height,
        depth_or_array_layers: 1,
    };

    let output_texture_extent = wgpu::Extent3d {
        width: puzzle_dimensions.piece_width,
        height: puzzle_dimensions.piece_height,
        depth_or_array_layers: puzzle_dimensions.num_pieces,
    };

    let src_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            // | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST,
        label: Some("piece-texture"),
    });
    let src_texture_view = src_texture.create_view(&Default::default());
    let src_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: output_texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING,
    });
    let output_texture_view = output_texture.create_view(&Default::default());

    // let piece_width = 168;
    // let piece_height = 168;

    println!("Generating control points...");
    let gen_start = Instant::now();
    let mut all_points = vec![];
    for _i in 0..puzzle_dimensions.num_pieces {
        let top_points = piece::Edge::gen_edge(
            puzzle_dimensions.piece_width - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut right_points = piece::Edge::gen_edge(
            puzzle_dimensions.piece_height - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut bottom_points = piece::Edge::gen_edge(
            puzzle_dimensions.piece_width - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut left_points = piece::Edge::gen_edge(
            puzzle_dimensions.piece_height - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );

        let mut control_points = top_points.points;
        control_points.append(&mut right_points.points);
        control_points.append(&mut bottom_points.points);
        control_points.append(&mut left_points.points);
        // control_points.push(control_points[0]);

        let spline = CatmullRomSpline::new(
            control_points
                .clone()
                .chunks(2)
                .map(|c| [c[0] as f64, c[1] as f64])
                .collect::<Vec<[f64; 2]>>(),
            0.5,
            true,
        );

        let step = 5;

        let mut curve_points = (0..(spline.control_points.len() - 1) * step)
            .map(|x| spline.sample(x as f64 / step as f64).unwrap().transpose())
            .flat_map(|points| [points[0] as f32, points[1] as f32])
            .collect::<Vec<f32>>();

        curve_points.push(curve_points[0]);
        curve_points.push(curve_points[1]);
        all_points.append(&mut curve_points);
    }

    println!(
        "Generated {} control points in {}µs",
        all_points.len(),
        gen_start.elapsed().as_micros()
    );

    let points_buffer_size = (all_points.len() * std::mem::size_of::<f32>()) as u64;

    let curve_points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: points_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM,
    });

    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(points_buffer_size),
                    },
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        });

    let mut texture_information_data =
        Vec::<i32>::with_capacity(puzzle_dimensions.num_pieces as usize * 4);
    for x in 0..puzzle_dimensions.pieces_x {
        for y in 0..puzzle_dimensions.pieces_y {
            let x_coord = (x * puzzle_dimensions.piece_width) as i32
                - (puzzle_dimensions.piece_padding * 2 * (x)) as i32
                - puzzle_dimensions.piece_padding as i32;
            let y_coord = (y * puzzle_dimensions.piece_height) as i32
                - (puzzle_dimensions.piece_padding * 2 * (y)) as i32
                - puzzle_dimensions.piece_padding as i32;
            let width = puzzle_dimensions.piece_width;
            let height = puzzle_dimensions.piece_height;
            texture_information_data.push(x_coord);
            texture_information_data.push(y_coord);
            texture_information_data.push(width as i32);
            texture_information_data.push(height as i32);
        }
    }

    let mask_output_size = wgpu::Extent3d {
        width: (puzzle_dimensions.padded_piece_width + puzzle_dimensions.piece_padding * 2)
            * puzzle_dimensions.pieces_x,
        height: (puzzle_dimensions.piece_height + puzzle_dimensions.piece_padding * 2)
            * puzzle_dimensions.pieces_y,
        depth_or_array_layers: 1,
    };

    println!("Mask output texture size {:?}", mask_output_size);

    let mask_output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mask output texture"),
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        mip_level_count: 1,
        sample_count: 1,
        size: mask_output_size,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
    });

    // let vertices = all_points
    //     .windows(3)
    //     .flat_map(|x| x.to_owned())
    //     .collect::<Vec<f32>>();

    let vertices = (0..puzzle_dimensions.num_pieces)
        .flat_map(|i| [0., 0., 0.5, 0., 0.5, 1.])
        .collect::<Vec<f32>>();

    let vertices_slice: &[u8] = bytemuck::cast_slice(&vertices);

    let uniform_alignment =
        device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;

    let aligned_vertices = unsafe {
        std::slice::from_raw_parts(
            vertices_slice.as_ptr(),
            vertices_slice.len() * uniform_alignment as usize,
        )
    };

    let mask_curves_vertices_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask curve vertices"),
            contents: vertices_slice,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
        });

    // let mask_curves_vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("Mask curve vertices"),
    //     size: aligned_vertices.len() as u64,
    //     mapped_at_creation: false,
    //     usage: wgpu::BufferUsages::VERTEX
    //         | wgpu::BufferUsages::UNIFORM
    //         | wgpu::BufferUsages::COPY_DST,
    // });

    let texture_information_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&texture_information_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: (output_buffer_row_size
            * puzzle_dimensions.piece_height
            * puzzle_dimensions.num_pieces) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let mask_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: (mask_output_size.width * mask_output_size.height * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let mask_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
            }],
        });

    // A pipeline specifies the operation of a shader

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_mask_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render masks pipeline layout"),
            bind_group_layouts: &[&mask_bind_group_layout],
            push_constant_ranges: &[],
        });

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Mask render pipeline"),
        layout: Some(&render_mask_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::default(),
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("src bind group"),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&src_texture_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&output_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: curve_points_buffer.as_entire_binding(),
            },
        ],
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: texture_information_buffer.as_entire_binding(),
        }],
    });

    let mask_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &mask_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &mask_curves_vertices_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(
                    vertices.len() as u64 / puzzle_dimensions.num_pieces as u64,
                ),
            }),
        }],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &src_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * puzzle_dimensions.padded_width),
            rows_per_image: std::num::NonZeroU32::new(puzzle_dimensions.height),
        },
        texture_extent,
    );

    queue.write_buffer(&curve_points_buffer, 0, bytemuck::cast_slice(&all_points));

    // queue.write_buffer(&mask_curves_vertices_buffer, 0, aligned_vertices);

    let mask_output_texture_view =
        mask_output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &mask_output_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&render_pipeline);
        let points_per_piece = vertices.len() as u32 / puzzle_dimensions.num_pieces;
        for i in 0..puzzle_dimensions.num_pieces {
            let offset = (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
            rpass.set_bind_group(0, &mask_bind_group, &[i * points_per_piece]);
            rpass.draw(0..points_per_piece, 0..1);
        }
    }

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_bind_group(1, &uniform_bind_group, &[]);

        cpass.dispatch(
            puzzle_dimensions.piece_width / 16,
            puzzle_dimensions.piece_height / 16,
            puzzle_dimensions.num_pieces,
        );

        // cpass.dispatch(puzzle_dimensions.pieces_x, puzzle_dimensions.pieces_y, 1)

        // for x in 0..pieces_x {
        //     for y in 0..pieces_y {
        //     }
        // }
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_texture_to_buffer(
        output_texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(output_buffer_row_size),
                rows_per_image: std::num::NonZeroU32::new(puzzle_dimensions.piece_height),
            },
        },
        output_texture_extent,
    );

    encoder.copy_texture_to_buffer(
        mask_output_texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &mask_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(mask_output_size.width * 4),
                rows_per_image: std::num::NonZeroU32::new(mask_output_size.height),
            },
        },
        mask_output_size,
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let buffer_future = slice.map_async(wgpu::MapMode::Read);

    let render_slice = mask_staging_buffer.slice(..);
    let render_buffer_future = render_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = render_buffer_future.await {
        let data = render_slice.get_mapped_range();
        let result: &[u8] = bytemuck::cast_slice(&data);

        let saving_start = Instant::now();
        println!("Saving images...");
        let mut image = image::RgbaImage::from_raw(
            mask_output_size.width,
            mask_output_size.height,
            result.to_vec(),
        )
        .unwrap();

        let path = format!("mask.png",);
        image.save(path).unwrap();

        println!(
            "Took {}ms to save images",
            saving_start.elapsed().as_millis()
        );
    } else {
        panic!("AHHH")
    }

    if let Ok(()) = buffer_future.await {
        let duration = start.elapsed();
        println!("Took {}ms to render and copy buffers", duration.as_millis());
        let data = slice.get_mapped_range();
        let result: &[u8] = bytemuck::cast_slice(&data);

        let saving_start = Instant::now();
        println!("Saving images...");
        result
            .par_chunks((output_buffer_row_size * puzzle_dimensions.piece_height) as usize)
            .enumerate()
            .for_each(|(i, raw_image)| {
                let mut new_image = image::RgbaImage::from_raw(
                    output_buffer_row_size / 4,
                    puzzle_dimensions.piece_height,
                    raw_image.to_vec(),
                )
                .unwrap();

                let cropped_image = image::imageops::crop(
                    &mut new_image,
                    0,
                    0,
                    puzzle_dimensions.piece_width,
                    puzzle_dimensions.piece_height,
                )
                .to_image();

                let path = format!("gpu-out/test-image{}.png", i);
                cropped_image.save(path).unwrap();
            });

        println!(
            "Took {}ms to save images",
            saving_start.elapsed().as_millis()
        );
    } else {
        panic!("AHHH")
    }

    Some(())
}

fn main() {
    let start = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }

    println!("\nFinished in {}ms.", start.elapsed().as_millis());
}
