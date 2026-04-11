#pragma once

#include "../include/hittable_list.h"
#include "../include/sphere.h"
#include "../include/material.h"
#include "../include/camera.h"
#include <memory>
#include <random>

// Builds the RTIOW Book 1 final scene: 484 random spheres + 3 feature spheres.
// Returns ownership of the world (HittableList) and fills cam with the standard
// RTIOW viewpoint settings.
// All Material objects are owned by `mats` which must outlive the returned world.
inline std::shared_ptr<HittableList> make_random_spheres_scene(
        CameraParams& cam,
        std::vector<std::shared_ptr<Material>>& mats) {

    auto world = std::make_shared<HittableList>();
    std::mt19937 rng(1337);  // fixed seed → deterministic scene

    // Ground
    auto mat_ground = std::make_shared<Lambertian>(color(0.5f, 0.5f, 0.5f));
    mats.push_back(mat_ground);
    world->add(std::make_shared<Sphere>(point3(0.f, -1000.f, 0.f), 1000.f,
                                        mat_ground.get()));

    // Random small spheres
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose = rand_float(rng);
            point3 center(a + 0.9f * rand_float(rng),
                          0.2f,
                          b + 0.9f * rand_float(rng));

            // Skip spheres too close to the three big ones
            if ((center - point3(4.f, 0.2f, 0.f)).length() <= 0.9f) continue;

            std::shared_ptr<Material> mat;
            if (choose < 0.8f) {
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

    // Three large feature spheres
    auto mat1 = std::make_shared<Dielectric>(1.5f);
    mats.push_back(mat1);
    world->add(std::make_shared<Sphere>(point3(0.f, 1.f, 0.f), 1.f, mat1.get()));

    auto mat2 = std::make_shared<Lambertian>(color(0.4f, 0.2f, 0.1f));
    mats.push_back(mat2);
    world->add(std::make_shared<Sphere>(point3(-4.f, 1.f, 0.f), 1.f, mat2.get()));

    auto mat3 = std::make_shared<Metal>(color(0.7f, 0.6f, 0.5f), 0.f);
    mats.push_back(mat3);
    world->add(std::make_shared<Sphere>(point3(4.f, 1.f, 0.f), 1.f, mat3.get()));

    // Camera settings for this scene
    cam.aspect_ratio      = 16.f / 9.f;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.vfov              = 20.f;
    cam.lookfrom          = point3(13.f, 2.f, 3.f);
    cam.lookat            = point3(0.f, 0.f, 0.f);
    cam.vup               = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle     = 0.6f;
    cam.focus_dist        = 10.f;
    cam.background        = color(0.5f, 0.7f, 1.f);  // sky blue

    return world;
}
