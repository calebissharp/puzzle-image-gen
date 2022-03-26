mod piece;
mod spline;
use image::GenericImage;
use image::{open, RgbaImage};
use spline::CatmullRomSpline;
use std::borrow::Cow;

fn find_closest_multiple(n: u32, x: u32) -> u32 {
    ((n - 1) | (x - 1)) + 1
}

async fn run() {
    execute_gpu().await;
}

async fn execute_gpu() -> Option<()> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let img = open("./test-images/uv.jpg").unwrap().to_rgba8();

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

    Some(())
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_img: RgbaImage,
) -> Option<()> {
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

    let pieces_x = 16;
    let pieces_y = 16;
    let num_pieces = pieces_x * pieces_y;
    let piece_padding = 16;

    let piece_width = dimensions.0 / pieces_x + piece_padding * 2;
    let piece_height = dimensions.1 / pieces_y + piece_padding * 2;

    let output_buffer_row_size = find_closest_multiple(piece_width * 4, 256);

    println!(
        "PIECE_WIDTH: {}, PIECE_HEIGHT: {}, NUM_PIECES_X: {}, NUM_PIECES_Y: {}",
        piece_width, piece_height, pieces_x, pieces_y,
    );

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            &include_str!("cookie-cutter.wgsl")
                .replace("$PIECE_WIDTH$", &piece_width.to_string())
                .replace("$PIECE_HEIGHT$", &piece_height.to_string())
                .replace("$NUM_PIECES_X$", &pieces_x.to_string())
                .replace("$NUM_PIECES_Y$", &pieces_y.to_string())
                .replace("$NUM_PIECES$", &num_pieces.to_string()),
        )),
    });

    let texture_extent = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let output_texture_extent = wgpu::Extent3d {
        width: piece_width,
        height: piece_height,
        depth_or_array_layers: num_pieces,
    };

    let src_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
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

    let piece_width = 168;
    let piece_height = 168;

    let top_points = piece::Edge::gen_edge(piece::Side::TOP, piece_width);
    let mut right_points = piece::Edge::gen_edge(piece::Side::RIGHT, piece_height);
    let mut bottom_points = piece::Edge::gen_edge(piece::Side::BOTTOM, piece_width);
    let mut left_points = piece::Edge::gen_edge(piece::Side::LEFT, piece_height);

    println!("Generating control points...");
    let mut control_points = top_points.points;
    control_points.append(&mut right_points.points);
    control_points.append(&mut bottom_points.points);
    control_points.append(&mut left_points.points);
    // control_points.push(control_points[0]);

    let spline = CatmullRomSpline::new(control_points.clone(), 0.5, true);

    let step = 10;

    let mut points = (0..(spline.control_points.len() - 1) * step)
        .map(|x| spline.sample(x as f64 / step as f64).unwrap().transpose())
        .flat_map(|points| [points[0] as f32, points[1] as f32])
        .collect::<Vec<f32>>();

    points.push(points[0]);
    points.push(points[1]);

    println!("Generated {} control points!", points.len());

    let points_buffer_size = (points.len() * std::mem::size_of::<f32>()) as u64;

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

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        mapped_at_creation: false,
        size: (output_buffer_row_size * piece_height * num_pieces) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    // A pipeline specifies the operation of a shader

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&texture_bind_group_layout],
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

    println!("bytes_per_row: {}", dimensions.0);

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
            bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
            rows_per_image: std::num::NonZeroU32::new(dimensions.1),
        },
        texture_extent,
    );

    queue.write_buffer(&curve_points_buffer, 0, bytemuck::cast_slice(&points));

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, num_pieces);
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
                rows_per_image: std::num::NonZeroU32::new(piece_height),
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
        let data = slice.get_mapped_range();
        let result: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        println!("Saving images...");
        for (i, raw_image) in result
            .chunks((output_buffer_row_size * piece_height) as usize)
            .enumerate()
        {
            let mut new_image = image::RgbaImage::from_raw(
                output_buffer_row_size / 4,
                piece_height,
                raw_image.to_vec(),
            )
            .unwrap();

            let cropped_image =
                image::imageops::crop(&mut new_image, 0, 0, piece_width, piece_height).to_image();

            let path = format!("gpu-out/test-image{}.png", i);
            cropped_image.save(path).unwrap();
        }
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
