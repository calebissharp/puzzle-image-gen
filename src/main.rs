mod piece;
mod spline;
mod util;

use core::num;
use image::GenericImage;
use image::{open, RgbaImage};
use log::{debug, error, info, warn};
use rayon::prelude::*;
use spline::CatmullRomSpline;
use std::borrow::Cow;
use std::time::Instant;
use util::find_closest_multiple;
use wgpu::util::DeviceExt;

async fn run() {
    execute_gpu().await;
}

struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
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
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, img).await;
    // draw_masks(&device, &queue).await;

    Some(())
}

async fn draw_masks(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<()> {
    let src_img_width = 1024;
    let src_img_height = 1024;
    let piece_padding = 32;
    let pieces_x = 32;
    let pieces_y = 32;
    let num_pieces = pieces_x * pieces_y;
    let img_width = pieces_x * piece_padding + src_img_width;
    let img_height = pieces_y * piece_padding + src_img_height;

    let buffer_dimensions = BufferDimensions::new(img_width, img_height);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_extent = wgpu::Extent3d {
        width: buffer_dimensions.width as u32,
        height: buffer_dimensions.height as u32,
        depth_or_array_layers: num_pieces as u32,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        label: None,
    });

    let command_buffer = {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        for x in 0..pieces_x {
            for y in 0..pieces_y {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &texture.create_view(&wgpu::TextureViewDescriptor {
                            base_array_layer: 1,
                            ..Default::default()
                        }),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });

                encoder.clear_texture(
                    &texture,
                    &wgpu::ImageSubresourceRange {
                        ..Default::default()
                    },
                )
            }
        }

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(buffer_dimensions.padded_bytes_per_row as u32)
                            .unwrap(),
                    ),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );

        encoder.finish()
    };

    queue.submit(Some(command_buffer));

    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        println!("result length {}", result.len());
        println!(
            "{}",
            buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height
        );

        println!("Saving mask image...");
        let image = image::RgbaImage::from_raw(
            buffer_dimensions.width as u32,
            buffer_dimensions.height as u32,
            result,
        )
        .unwrap();

        image.save("mask.png").unwrap();
    }

    Some(())
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_img: RgbaImage,
) -> Option<()> {
    let start = std::time::Instant::now();

    let original_image_dimensions = src_img.dimensions();
    // Image width needs to be a multiple of 256 in order
    // for copy_texture_to_buffer to work
    let mut image = RgbaImage::new(
        find_closest_multiple(src_img.width(), 256),
        src_img.height(),
    );
    println!(
        "Padding image for GPU\nOriginal: ({}x{}), New: ({}x{})",
        src_img.width(),
        src_img.height(),
        image.width(),
        image.height(),
    );
    image.copy_from(&src_img, 0, 0).unwrap();

    let dimensions = image.dimensions();

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
    let mut all_points = vec![];
    for i in 0..puzzle_dimensions.num_pieces {
        let top_points = piece::Edge::gen_edge(
            piece::Side::TOP,
            puzzle_dimensions.piece_width - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut right_points = piece::Edge::gen_edge(
            piece::Side::RIGHT,
            puzzle_dimensions.piece_height - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut bottom_points = piece::Edge::gen_edge(
            piece::Side::BOTTOM,
            puzzle_dimensions.piece_width - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );
        let mut left_points = piece::Edge::gen_edge(
            piece::Side::LEFT,
            puzzle_dimensions.piece_height - puzzle_dimensions.piece_padding * 2,
            puzzle_dimensions.piece_padding,
        );

        let mut control_points = top_points.points;
        control_points.append(&mut right_points.points);
        control_points.append(&mut bottom_points.points);
        control_points.append(&mut left_points.points);
        // control_points.push(control_points[0]);

        let spline = CatmullRomSpline::new(control_points.clone(), 0.5, true);

        let step = 5;

        let mut curve_points = (0..(spline.control_points.len() - 1) * step)
            .map(|x| spline.sample(x as f64 / step as f64).unwrap().transpose())
            .flat_map(|points| [points[0] as f32, points[1] as f32])
            .collect::<Vec<f32>>();

        curve_points.push(curve_points[0]);
        curve_points.push(curve_points[1]);
        all_points.append(&mut curve_points);
    }

    println!("Generated {} control points!", all_points.len());

    println!("{}", all_points[0]);

    let points_buffer_size = (all_points.len() * std::mem::size_of::<f32>()) as u64;

    let curve_points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: points_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

    println!("{:?}", &texture_information_data[0..8]);

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

    // A pipeline specifies the operation of a shader

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
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

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let buffer_future = slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let duration = start.elapsed();
        println!("Took {}ms to generate", duration.as_millis());
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
}
