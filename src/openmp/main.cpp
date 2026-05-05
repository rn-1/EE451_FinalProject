// OpenMP ray tracer — CPU multi-threaded implementation.
// Each scanline row is assigned to a thread; each thread has its own RNG.
// Usage:
//   OMP_NUM_THREADS=N ./bin/openmp_rt --scene [random|cornell] --width W
//                                     --height H --spp N [--depth D]
//                                     --output path.ppm [--timing-only]

#include "../../include/vec3.h"
#include "../../include/ray.h"
#include "../../include/interval.h"
#include "../../include/utils.h"
#include "../../include/hittable.h"
#include "../../include/hittable_list.h"
#include "../../include/sphere.h"
#include "../../include/quad.h"
#include "../../include/material.h"
#include "../../include/camera.h"
#include "../../scenes/random_spheres.h"
#include "../../scenes/cornell_box.h"
#include "../../scenes/simple.h"
#include "../../scenes/medium.h"
#include "../../scenes/complex.h"

#include <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

int main(int argc, char** argv) {
    // --- Parse arguments ---
    std::string scene_name  = get_arg(argc, argv, "--scene",  "random");
    int         img_width   = std::stoi(get_arg(argc, argv, "--width",  "400"));
    int         spp         = std::stoi(get_arg(argc, argv, "--spp",    "100"));
    int         depth       = std::stoi(get_arg(argc, argv, "--depth",  "50"));
    std::string output_path = get_arg(argc, argv, "--output", "output/openmp_out.ppm");
    bool        timing_only = has_flag(argc, argv, "--timing-only");

    // --- Build scene ---
    CameraParams cam;
    std::vector<std::shared_ptr<Material>> mats;
    std::shared_ptr<HittableList> world;

    if (scene_name == "cornell") {
        world = make_cornell_box_scene(cam, mats);
    } else if (scene_name == "simple") {
        world = make_simple_scene(cam, mats);
    } else if (scene_name == "medium") {
        world = make_medium_scene(cam, mats);
    } else if (scene_name == "complex") {
        world = make_complex_scene(cam, mats);
    }  else {
        world = make_random_spheres_scene(cam, mats);
    }

    cam.image_width       = img_width;
    cam.samples_per_pixel = spp;
    cam.max_depth         = depth;
    cam.initialize();

    int nthreads = omp_get_max_threads();

    if (!timing_only) {
        std::cerr << "OpenMP (" << nthreads << " threads)"
                  << " | scene=" << scene_name
                  << " res=" << cam.image_width << "x" << cam.image_height
                  << " spp=" << spp << " depth=" << depth << "\n";
    }

    // --- Allocate framebuffer ---
    int total_pixels = cam.image_height * cam.image_width;
    std::vector<color> fb(total_pixels, color(0.f, 0.f, 0.f));
    long long traced_rays = 0;

    // --- Render (timed) ---
    Timer timer;
    timer.begin();

    // Parallel over rows; each thread gets its own mt19937 seeded uniquely.
    // schedule(dynamic,4) helps balance load when some rows have more
    // expensive rays (e.g. many sphere hits vs. sky misses).
    #pragma omp parallel reduction(+:traced_rays) default(none) shared(cam, world, fb)
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(42u + static_cast<unsigned>(tid) * 1000003u);
        long long local_rays = 0;

        #pragma omp for schedule(dynamic, 4)
        for (int row = 0; row < cam.image_height; ++row) {
            for (int col = 0; col < cam.image_width; ++col) {
                color pixel(0.f, 0.f, 0.f);
                for (int s = 0; s < cam.samples_per_pixel; ++s) {
                    ray r = get_ray(cam, col, row, rng);
                    pixel += ray_color(r, cam.max_depth, *world, cam.background, rng, &local_rays);
                }
                fb[row * cam.image_width + col] = pixel / (float)cam.samples_per_pixel;
            }
        }

        traced_rays += local_rays;
    }

    double ms = timer.elapsed_ms();
    double rays_per_sec = traced_rays / (ms / 1000.0);

    if (timing_only) {
        std::cout << ms << "\t" << traced_rays << "\t" << rays_per_sec << "\n";
    } else {
        std::cerr << "Render time: " << ms << " ms\n";
        std::cerr << "Traced rays: " << traced_rays << "\n";
        std::cerr << "Throughput: " << rays_per_sec / 1e9 << " Grays/s\n";
        write_ppm(output_path, fb, cam.image_width, cam.image_height);
        std::cerr << "Output written to: " << output_path << "\n";
    }

    return 0;
}
