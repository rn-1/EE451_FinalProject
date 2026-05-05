#pragma once

#include "../include/hittable_list.h"
#include "../include/sphere.h"
#include "../include/material.h"
#include "../include/camera.h"
#include <memory>
#include <vector>

// simple scene w one floating sphere
inline std::shared_ptr<HittableList> make_complex_scene(
        CameraParams& cam,
        std::vector<std::shared_ptr<Material>>& mats) {

    auto world = std::make_shared<HittableList>();
    std::mt19937 rng(1337);

    // Materials
    auto lavender = std::make_shared<Lambertian>(color(0.59f, 0.48f, 0.71f));
    auto ground = std::make_shared<Metal>(color(0.75f, 0.75f, 0.75f), 0.f);
    mats.insert(mats.end(), {lavender, ground});

    for (int a = -5; a < 5; ++a) {
        for (int b = -5; b < 5; ++b) {
            float choose = rand_float(rng);
            point3 center(a + 0.9f * rand_float(rng),
                          0.2f,
                          b + 0.9f * rand_float(rng));

            // Skip spheres too close to the three big ones
            if ((center - point3(4.f, 0.2f, 0.f)).length() <= 0.9f) continue;

            std::shared_ptr<Material> mat;
            if (choose < 0.6f) {
                color alb = random_vec(rng) * random_vec(rng);
                mat = std::make_shared<Lambertian>(alb);
            } else if (choose < 0.95f) {
                color alb = random_vec(rng, 0.5f, 1.f);
                float fuzz = rand_float(rng, 0.f, 0.5f);
                mat = std::make_shared<Metal>(alb, fuzz);
            } else {
                mat = std::make_shared<Dielectric>(1.5f);
            }
            mats.push_back(mat);
            world->add(std::make_shared<Sphere>(center, 0.2f, mat.get()));
        }
    }

    world->add(std::make_shared<Sphere>(point3(0,0,-1), 0.5, lavender.get()));
    world->add(std::make_shared<Sphere>(point3(0,-100.5,-1), 100, ground.get()));

    cam.aspect_ratio      = 16.f / 9.f;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.vfov              = 20.f;
    cam.lookfrom          = point3(0.f, 2.5f, 10.f);
    cam.lookat            = point3(0.f, 0.f, 0.f);
    cam.vup               = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle     = 0.6f;
    cam.focus_dist        = 10.f;
    cam.background        = color(0.7f, 0.8f, 1.0f);

    return world;
}
