#pragma once

#include "../include/hittable_list.h"
#include "../include/sphere.h"
#include "../include/material.h"
#include "../include/camera.h"
#include <memory>
#include <vector>

// simple scene w one floating sphere
inline std::shared_ptr<HittableList> make_simple_scene(
        CameraParams& cam,
        std::vector<std::shared_ptr<Material>>& mats) {

    auto world = std::make_shared<HittableList>();

    // Materials
    auto lavender = std::make_shared<Lambertian>(color(0.59f, 0.48f, 0.71f));
    auto ground = std::make_shared<Lambertian>(color(0.17f, 0.15f, 0.17f));
    mats.insert(mats.end(), {lavender, ground});

    world->add(std::make_shared<Sphere>(point3(0,0,-1), 0.5, lavender.get()));
    world->add(std::make_shared<Sphere>(point3(0,-100.5,-1), 100, ground.get()));

    cam.aspect_ratio      = 16.f / 9.f;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.vfov              = 30.f;
    cam.lookfrom          = point3(0.f, 2.5f, 10.f);
    cam.lookat            = point3(0.f, 0.f, 0.f);
    cam.vup               = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle     = 0.6f;
    cam.focus_dist        = 10.f;
    cam.background        = color(0.7f, 0.8f, 1.0f);

    return world;
}
