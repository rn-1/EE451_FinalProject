#pragma once

#include "ray.h"
#include "hittable.h"
#include "interval.h"
#include "utils.h"
#include <cmath>

#ifndef __CUDACC__
#include "material.h"
#include <vector>
#include <random>
#endif

// Camera parameters — plain-old-data so it can be copied to GPU
struct CameraParams {
    // --- User-specified ---
    float  aspect_ratio     = 16.f / 9.f;
    int    image_width      = 400;
    int    samples_per_pixel = 100;
    int    max_depth        = 50;
    float  vfov             = 90.f;    // vertical FOV degrees
    point3 lookfrom         = point3(0, 0, 0);
    point3 lookat           = point3(0, 0, -1);
    vec3   vup              = vec3(0, 1, 0);
    float  defocus_angle    = 0.f;     // 0 = no depth of field
    float  focus_dist       = 10.f;
    color  background       = color(0.7f, 0.8f, 1.0f); // sky gradient fallback

    // --- Derived (filled by initialize()) ---
    int    image_height     = 0;
    point3 center;
    point3 pixel00_loc;
    vec3   pixel_delta_u;
    vec3   pixel_delta_v;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;

    void initialize() {
        image_height = std::max(1, (int)(image_width / aspect_ratio));

        center = lookfrom;

        float theta      = deg_to_rad(vfov);
        float h          = tanf(theta / 2.f);
        float vp_height  = 2.f * h * focus_dist;
        float vp_width   = vp_height * ((float)image_width / image_height);

        vec3 w_ = unit_vector(lookfrom - lookat);
        vec3 u_ = unit_vector(cross(vup, w_));
        vec3 v_ = cross(w_, u_);

        vec3 vp_u = vp_width  * u_;
        vec3 vp_v = vp_height * (-v_);

        pixel_delta_u = vp_u / (float)image_width;
        pixel_delta_v = vp_v / (float)image_height;

        point3 vp_upper_left = center - focus_dist * w_
                                      - vp_u / 2.f - vp_v / 2.f;
        pixel00_loc = vp_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        float defocus_radius = focus_dist * tanf(deg_to_rad(defocus_angle / 2.f));
        defocus_disk_u = u_ * defocus_radius;
        defocus_disk_v = v_ * defocus_radius;
    }
};

#ifndef __CUDACC__
// ---- CPU-side ray generation & render ----

inline ray get_ray(const CameraParams& cam, int col, int row,
                   std::mt19937& rng) {
    // Stratified sampling: random offset within pixel
    float offset_u = rand_float(rng) - 0.5f;
    float offset_v = rand_float(rng) - 0.5f;

    point3 pixel_center = cam.pixel00_loc
                        + (col + offset_u) * cam.pixel_delta_u
                        + (row + offset_v) * cam.pixel_delta_v;

    point3 ray_origin;
    if (cam.defocus_angle <= 0.f) {
        ray_origin = cam.center;
    } else {
        vec3 p = random_in_unit_disk(rng);
        ray_origin = cam.center + p.x() * cam.defocus_disk_u
                                + p.y() * cam.defocus_disk_v;
    }
    return ray(ray_origin, pixel_center - ray_origin);
}

// Recursive ray color (serial and OpenMP)
inline color ray_color(const ray& r, int depth,
                       const Hittable& world,
                       const color& background,
                       std::mt19937& rng) {
    if (depth <= 0) return color(0.f, 0.f, 0.f);

    hit_record rec;
    if (!world.hit(r, interval(0.001f, INF), rec)) {
        // Sky background: gradient when no hit
        // For Cornell Box, background is black (0,0,0)
        if (background.x() == 0.f && background.y() == 0.f && background.z() == 0.f) {
            return background;
        }
        vec3 unit = unit_vector(r.direction());
        float t = 0.5f * (unit.y() + 1.f);
        return (1.f - t) * color(1.f, 1.f, 1.f) + t * background;
    }

    color emitted = rec.mat->emitted();

    color attenuation;
    ray   scattered;
    if (!rec.mat->scatter(r, rec, attenuation, scattered, rng))
        return emitted;

    return emitted + attenuation * ray_color(scattered, depth - 1, world, background, rng);
}

// Full render — fills framebuffer[height * width]
// rng_seed: base seed; thread offset applied in OpenMP version
inline void render(const CameraParams& cam, const Hittable& world,
                   std::vector<color>& fb, unsigned int rng_seed = 42) {
    fb.resize(cam.image_height * cam.image_width);
    std::mt19937 rng(rng_seed);

    for (int row = 0; row < cam.image_height; ++row) {
        for (int col = 0; col < cam.image_width; ++col) {
            color pixel(0.f, 0.f, 0.f);
            for (int s = 0; s < cam.samples_per_pixel; ++s) {
                ray r = get_ray(cam, col, row, rng);
                pixel += ray_color(r, cam.max_depth, world, cam.background, rng);
            }
            fb[row * cam.image_width + col] = pixel / (float)cam.samples_per_pixel;
        }
    }
}
#endif // !__CUDACC__
