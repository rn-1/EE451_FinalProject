#pragma once

#include "../include/hittable_list.h"
#include "../include/quad.h"
#include "../include/sphere.h"
#include "../include/material.h"
#include "../include/camera.h"
#include <memory>
#include <vector>

// Classic Cornell Box scene (quads + emissive light).
// Fills cam with the standard Cornell Box viewpoint.
inline std::shared_ptr<HittableList> make_cornell_box_scene(
        CameraParams& cam,
        std::vector<std::shared_ptr<Material>>& mats) {

    auto world = std::make_shared<HittableList>();

    // Materials
    auto red   = std::make_shared<Lambertian>(color(0.65f, 0.05f, 0.05f));
    auto white = std::make_shared<Lambertian>(color(0.73f, 0.73f, 0.73f));
    auto green = std::make_shared<Lambertian>(color(0.12f, 0.45f, 0.15f));
    auto light = std::make_shared<DiffuseLight>(color(15.f, 15.f, 15.f));
    mats.insert(mats.end(), {red, white, green, light});

    // Walls (555 x 555 x 555 box)
    // Left wall  (red, x=555 plane, facing -x)
    world->add(std::make_shared<Quad>(
        point3(555.f, 0.f, 0.f), vec3(0.f, 555.f, 0.f), vec3(0.f, 0.f, 555.f),
        red.get()));
    // Right wall (green, x=0 plane, facing +x)
    world->add(std::make_shared<Quad>(
        point3(0.f, 0.f, 0.f), vec3(0.f, 555.f, 0.f), vec3(0.f, 0.f, 555.f),
        green.get()));
    // Ceiling light
    world->add(std::make_shared<Quad>(
        point3(213.f, 554.f, 227.f), vec3(130.f, 0.f, 0.f), vec3(0.f, 0.f, 105.f),
        light.get()));
    // Floor
    world->add(std::make_shared<Quad>(
        point3(0.f, 0.f, 0.f), vec3(555.f, 0.f, 0.f), vec3(0.f, 0.f, 555.f),
        white.get()));
    // Ceiling
    world->add(std::make_shared<Quad>(
        point3(555.f, 555.f, 555.f), vec3(-555.f, 0.f, 0.f), vec3(0.f, 0.f, -555.f),
        white.get()));
    // Back wall
    world->add(std::make_shared<Quad>(
        point3(0.f, 0.f, 555.f), vec3(555.f, 0.f, 0.f), vec3(0.f, 555.f, 0.f),
        white.get()));

    // Two boxes represented as sphere stand-ins using 5 quads each.
    // Box 1: tall box (approx 165x330x165, tilted ~15 deg)
    // We implement boxes as 6 quads manually.

    auto add_box = [&](const point3& pmin, const point3& pmax, Material* mat) {
        // 6 faces
        // front (z=pmax.z)
        world->add(std::make_shared<Quad>(
            point3(pmin.x(), pmin.y(), pmax.z()),
            vec3(pmax.x()-pmin.x(), 0, 0),
            vec3(0, pmax.y()-pmin.y(), 0), mat));
        // back (z=pmin.z)
        world->add(std::make_shared<Quad>(
            point3(pmax.x(), pmin.y(), pmin.z()),
            vec3(pmin.x()-pmax.x(), 0, 0),
            vec3(0, pmax.y()-pmin.y(), 0), mat));
        // left (x=pmin.x)
        world->add(std::make_shared<Quad>(
            point3(pmin.x(), pmin.y(), pmin.z()),
            vec3(0, 0, pmax.z()-pmin.z()),
            vec3(0, pmax.y()-pmin.y(), 0), mat));
        // right (x=pmax.x)
        world->add(std::make_shared<Quad>(
            point3(pmax.x(), pmin.y(), pmax.z()),
            vec3(0, 0, pmin.z()-pmax.z()),
            vec3(0, pmax.y()-pmin.y(), 0), mat));
        // top (y=pmax.y)
        world->add(std::make_shared<Quad>(
            point3(pmin.x(), pmax.y(), pmin.z()),
            vec3(pmax.x()-pmin.x(), 0, 0),
            vec3(0, 0, pmax.z()-pmin.z()), mat));
        // bottom (y=pmin.y)
        world->add(std::make_shared<Quad>(
            point3(pmin.x(), pmin.y(), pmax.z()),
            vec3(pmax.x()-pmin.x(), 0, 0),
            vec3(0, 0, pmin.z()-pmax.z()), mat));
    };

    // Short box
    add_box(point3(130.f, 0.f, 65.f), point3(295.f, 165.f, 230.f), white.get());
    // Tall box
    add_box(point3(265.f, 0.f, 295.f), point3(430.f, 330.f, 460.f), white.get());

    // Camera — looking into box from z=-800
    cam.aspect_ratio      = 1.f;
    cam.image_width       = 400;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.vfov              = 40.f;
    cam.lookfrom          = point3(278.f, 278.f, -800.f);
    cam.lookat            = point3(278.f, 278.f, 0.f);
    cam.vup               = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle     = 0.f;
    cam.background        = color(0.f, 0.f, 0.f);  // black — lighting only from emissive

    return world;
}
