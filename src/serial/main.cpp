// Serial ray tracer — single-threaded CPU implementation.
// Usage:
//   ./bin/serial_rt --scene [random|cornell] --width W --height H
//                   --spp N [--depth D] --output path.ppm [--timing-only]

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
    std::string output_path = get_arg(argc, argv, "--output", "output/serial_out.ppm");
    bool        timing_only = has_flag(argc, argv, "--timing-only");

    // --- Build scene ---
    CameraParams cam;
    std::vector<std::shared_ptr<Material>> mats;  // lifetime owner
    std::shared_ptr<HittableList> world;

    if (scene_name == "cornell") {
        world = make_cornell_box_scene(cam, mats);
    } else {
        world = make_random_spheres_scene(cam, mats);
    }

    // Override resolution / SPP from CLI
    cam.image_width       = img_width;
    cam.samples_per_pixel = spp;
    cam.max_depth         = depth;
    cam.initialize();

    if (!timing_only) {
        std::cerr << "Serial | scene=" << scene_name
                  << " res=" << cam.image_width << "x" << cam.image_height
                  << " spp=" << spp << " depth=" << depth << "\n";
    }

    // --- Render (timed) ---
    std::vector<color> fb;
    fb.resize(cam.image_height * cam.image_width);
    std::mt19937 rng(42);

    Timer timer;
    timer.begin();

    for (int row = 0; row < cam.image_height; ++row) {
        for (int col = 0; col < cam.image_width; ++col) {
            color pixel(0.f, 0.f, 0.f);
            for (int s = 0; s < cam.samples_per_pixel; ++s) {
                ray r = get_ray(cam, col, row, rng);
                pixel += ray_color(r, cam.max_depth, *world, cam.background, rng);
            }
            fb[row * cam.image_width + col] = pixel / (float)cam.samples_per_pixel;
        }
    }

    double ms = timer.elapsed_ms();

    if (timing_only) {
        std::cout << ms << "\n";
    } else {
        std::cerr << "Render time: " << ms << " ms\n";
        write_ppm(output_path, fb, cam.image_width, cam.image_height);
        std::cerr << "Output written to: " << output_path << "\n";
    }

    return 0;
}
