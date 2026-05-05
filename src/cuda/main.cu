// CUDA ray tracer — GPU-accelerated host entry point.
// Usage:
//   ./bin/cuda_rt --scene [random|cornell] --width W --height H
//                --spp N [--depth D] --output path.ppm [--timing-only]
//                [--device N]

#include "render.cuh"
#include "device_scene.cuh"
#include "../../include/utils.h"
#include "../../include/camera.h"

#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    // --- Parse arguments ---
    std::string scene_name  = get_arg(argc, argv, "--scene",  "random");
    int         img_width   = std::stoi(get_arg(argc, argv, "--width",  "400"));
    int         spp         = std::stoi(get_arg(argc, argv, "--spp",    "100"));
    int         depth       = std::stoi(get_arg(argc, argv, "--depth",  "50"));
    std::string output_path = get_arg(argc, argv, "--output", "output/cuda_out.ppm");
    int         device_id   = std::stoi(get_arg(argc, argv, "--device", "0"));
    bool        timing_only = has_flag(argc, argv, "--timing-only");

    // --- Select GPU ---
    cudaSetDevice(device_id);
    if (!timing_only) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        std::cerr << "GPU: " << prop.name
                  << " | SM count: " << prop.multiProcessorCount << "\n";
    }

    // --- Camera params (scene-specific defaults) ---
    CameraParams cam;

    if (scene_name == "cornell") {
        // Cornell Box defaults
        cam.aspect_ratio      = 1.f;
        cam.vfov              = 40.f;
        cam.lookfrom          = point3(278.f, 278.f, -800.f);
        cam.lookat            = point3(278.f, 278.f, 0.f);
        cam.vup               = vec3(0.f, 1.f, 0.f);
        cam.defocus_angle     = 0.f;
        cam.focus_dist        = 10.f;
        cam.background        = color(0.f, 0.f, 0.f);
    } else if (scene_name == "random") {
        // Random spheres defaults
        cam.aspect_ratio      = 16.f / 9.f;
        cam.vfov              = 20.f;
        cam.lookfrom          = point3(13.f, 2.f, 3.f);
        cam.lookat            = point3(0.f, 0.f, 0.f);
        cam.vup               = vec3(0.f, 1.f, 0.f);
        cam.defocus_angle     = 0.6f;
        cam.focus_dist        = 10.f;
        cam.background        = color(0.5f, 0.7f, 1.f);
    } else {
        cam.aspect_ratio      = 16.f / 9.f;
        cam.vfov              = 20.f;
        cam.lookfrom          = point3(0.f, 2.5f, 10.f);
        cam.lookat            = point3(0.f, 0.f, 0.f);
        cam.vup               = vec3(0.f, 1.f, 0.f);
        cam.defocus_angle     = 0.6f;
        cam.focus_dist        = 10.f;
        cam.background        = color(0.7f, 0.8f, 1.0f);
    }

    cam.image_width       = img_width;
    cam.samples_per_pixel = spp;
    cam.max_depth         = depth;
    cam.initialize();

    if (!timing_only) {
        std::cerr << "CUDA | scene=" << scene_name
                  << " res=" << cam.image_width << "x" << cam.image_height
                  << " spp=" << spp << " depth=" << depth << "\n";
    }

    // --- Build device scene ---
    DeviceScene scene;
    if (scene_name == "cornell") {
        scene = build_device_cornell_box();
    } else if (scene_name == "simple") {
        scene = build_device_simple_scene();
    } else if (scene_name == "medium") {
        scene = build_device_medium_scene();
    } else if (scene_name == "complex") {
        scene = build_device_complex_scene();
    }  else {
        scene = build_device_random_spheres();
    }

    // --- Render ---
    int N = cam.image_width * cam.image_height;
    std::vector<color> h_fb(N);

    float  kernel_ms = 0.f;
    double total_ms  = 0.0;
    unsigned long long traced_rays = 0ULL;

    cuda_render(cam, scene, h_fb.data(), &kernel_ms, &total_ms, &traced_rays);

    // --- Output ---
    double rays_per_sec = traced_rays / (total_ms / 1000.0);

    if (timing_only) {
        // Print: kernel_ms total_ms ray_count rays_per_sec (tab-separated)
        std::cout << kernel_ms << "\t" << total_ms << "\t" << traced_rays << "\t" << rays_per_sec << "\n";
    } else {
        std::cerr << "Render kernel:  " << kernel_ms << " ms\n";
        std::cerr << "Total GPU time: " << total_ms  << " ms\n";
        std::cerr << "Total rays: " << traced_rays << "\n";
        std::cerr << "Throughput: " << rays_per_sec / 1e9 << " Grays/s\n";
        write_ppm(output_path, h_fb, cam.image_width, cam.image_height);
        std::cerr << "Output written to: " << output_path << "\n";
    }

    free_device_scene(scene);
    return 0;
}
