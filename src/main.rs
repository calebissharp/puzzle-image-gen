mod piece;
mod spline;
mod util;

use bytemuck::{Pod, Zeroable};
use image::GenericImage;
use image::{open, RgbaImage};
use nalgebra::Orthographic3;
use piece::Puzzle;
use std::borrow::Cow;
use std::f32;
use std::mem;
use std::time::Instant;
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

    draw_masks(&device, &queue, img).await;

    Some(())
}

async fn draw_masks(device: &wgpu::Device, queue: &wgpu::Queue, src_img: RgbaImage) -> Option<()> {
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
        for i in 0..(puzzle.dimensions.num_pieces as usize) {
            let offset = (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
            rpass.set_bind_group(1, &local_bind_group, &[offset]);

            let buffer_size = buffer_sizes[i] as u64;
            let slice = buffer.slice(
                (buffer_offsets[i] as u64 * 4)
                    ..(buffer_offsets[i] as u64 * 4 + buffer_size * 4 as u64),
            );

            rpass.set_vertex_buffer(0, slice);
            rpass.draw(0..(buffer_size / 4) as u32, 0..1);
        }
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
        let image = image::RgbaImage::from_raw(
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
