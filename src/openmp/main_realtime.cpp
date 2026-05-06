// OpenMP real-time ray tracer with GLFW window.
// Usage: OMP_NUM_THREADS=N ./bin/openmp_realtime_rt [--scene random|cornell]
//                           [--width W] [--spp N] [--depth D]
// Defaults: 640x360, 1 spp, 3 depth.  ESC or close to quit.

#include <GLFW/glfw3.h>
#include <complex.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
    std::string scene_name = get_arg(argc, argv, "--scene", "random");
    int img_width = std::stoi(get_arg(argc, argv, "--width",  "640"));
    int spp       = std::stoi(get_arg(argc, argv, "--spp",    "1"));
    int depth     = std::stoi(get_arg(argc, argv, "--depth",  "3"));

    // --- Scene ---
    CameraParams cam;
    std::vector<std::shared_ptr<Material>> mats;
    std::shared_ptr<HittableList> world;
    if (scene_name == "cornell")
        world = make_cornell_box_scene(cam, mats);
    else if (scene_name == "complex")
        world = make_complex_scene(cam, mats);
    else
        world = make_random_spheres_scene(cam, mats);

    cam.image_width       = img_width;
    cam.samples_per_pixel = spp;
    cam.max_depth         = depth;
    cam.initialize();
    int W = cam.image_width, H = cam.image_height;

    // --- GLFW ---
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(W, H, "Ray Tracer (OpenMP)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // --- GL texture ---
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,1,0,1,-1,1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();

    std::vector<color>   fb(W * H);
    std::vector<uint8_t> rgba(W * H * 4);

    // Camera roll state
    vec3  base_vup   = cam.vup;
    vec3  fwd        = unit_vector(cam.lookat - cam.lookfrom);
    float roll_speed = 0.3f;

    int nthreads = omp_get_max_threads();
    fprintf(stderr, "OpenMP (%d threads) | scene=%s  res=%dx%d  spp=%d  depth=%d  | ESC to quit\n",
            nthreads, scene_name.c_str(), W, H, spp, depth);

    auto t0      = std::chrono::steady_clock::now();
    auto t_start = t0;
    int  frames  = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        // Camera roll
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        float a = (float)elapsed * roll_speed;
        float c = cosf(a), sr = sinf(a);
        cam.vup = base_vup * c
                + cross(fwd, base_vup) * sr
                + fwd * dot(fwd, base_vup) * (1.f - c);
        cam.initialize();

        // Render — each thread gets its own RNG seeded by frame + tid
        #pragma omp parallel default(none) shared(cam, world, fb, frames, depth, spp, W, H)
        {
            int tid = omp_get_thread_num();
            std::mt19937 rng(42u + (unsigned)frames * 1000003u
                                 + (unsigned)tid    * 999983u);
            #pragma omp for schedule(dynamic, 4)
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    color pixel(0.f, 0.f, 0.f);
                    for (int s = 0; s < spp; ++s) {
                        ray r = get_ray(cam, col, row, rng);
                        pixel += ray_color(r, depth, *world, cam.background, rng);
                    }
                    fb[row * W + col] = pixel / (float)spp;
                }
            }
        }

        // Convert float3 → RGBA and upload
        for (int i = 0; i < W * H; ++i) {
            rgba[4*i+0] = (uint8_t)to_byte(fb[i].x());
            rgba[4*i+1] = (uint8_t)to_byte(fb[i].y());
            rgba[4*i+2] = (uint8_t)to_byte(fb[i].z());
            rgba[4*i+3] = 255u;
        }
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
          glTexCoord2f(0.f,0.f); glVertex2f(0.f,1.f);
          glTexCoord2f(1.f,0.f); glVertex2f(1.f,1.f);
          glTexCoord2f(1.f,1.f); glVertex2f(1.f,0.f);
          glTexCoord2f(0.f,1.f); glVertex2f(0.f,0.f);
        glEnd();

        glfwSwapBuffers(window);

        ++frames;
        auto t1  = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s >= 0.5) {
            char title[128];
            snprintf(title, sizeof(title),
                     "Ray Tracer (OpenMP %d threads)  |  %.2f FPS  |  spp=%d  depth=%d  |  %dx%d",
                     nthreads, frames / s, spp, depth, W, H);
            glfwSetWindowTitle(window, title);
            frames = 0; t0 = t1;
        }
    }

    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
