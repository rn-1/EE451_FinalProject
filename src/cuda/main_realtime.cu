// Real-time GLFW ray tracer.
// Usage: ./bin/realtime_rt [--scene random|cornell] [--width W]
//                          [--spp N] [--depth D] [--device N]
// Default: 1280x720, 4 spp, 4 depth.  ESC or close window to quit.

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "render.cuh"
#include "device_scene.cuh"
#include "../../include/utils.h"
#include "../../include/camera.h"

int main(int argc, char** argv) {
    std::string scene_name = get_arg(argc, argv, "--scene",  "random");
    int img_width = std::stoi(get_arg(argc, argv, "--width",  "1280"));
    int spp       = std::stoi(get_arg(argc, argv, "--spp",    "4"));
    int depth     = std::stoi(get_arg(argc, argv, "--depth",  "4"));
    int device_id = std::stoi(get_arg(argc, argv, "--device", "0"));

    cudaSetDevice(device_id);

    // --- Camera ---
    CameraParams cam;
    if (scene_name == "cornell") {
        cam.aspect_ratio  = 1.f;
        cam.vfov          = 40.f;
        cam.lookfrom      = point3(278.f, 278.f, -800.f);
        cam.lookat        = point3(278.f, 278.f, 0.f);
        cam.vup           = vec3(0.f, 1.f, 0.f);
        cam.defocus_angle = 0.f;
        cam.focus_dist    = 10.f;
        cam.background    = color(0.f, 0.f, 0.f);
    } else {
        cam.aspect_ratio  = 16.f / 9.f;
        cam.vfov          = 20.f;
        cam.lookfrom      = point3(13.f, 2.f, 3.f);
        cam.lookat        = point3(0.f, 0.f, 0.f);
        cam.vup           = vec3(0.f, 1.f, 0.f);
        cam.defocus_angle = 0.6f;
        cam.focus_dist    = 10.f;
        cam.background    = color(0.5f, 0.7f, 1.f);
    }
    cam.image_width       = img_width;
    cam.samples_per_pixel = spp;
    cam.max_depth         = depth;
    cam.initialize();
    int W = cam.image_width;
    int H = cam.image_height;

    // --- Scene ---
    DeviceScene scene;
    if (scene_name == "cornell")
        scene = build_device_cornell_box();
    else
        scene = build_device_random_spheres();

    // --- GLFW window ---
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(W, H, "Ray Tracer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // no vsync — measure true FPS

    // --- OpenGL texture ---
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Ortho 2D projection (NDC 0..1)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // --- Host RGBA staging buffer ---
    std::vector<uchar4> h_rgba(W * H);

    // --- Persistent GPU state ---
    RenderState rs;
    cuda_render_init(rs, W, H);

    fprintf(stderr, "RTX 3060 | scene=%s  res=%dx%d  spp=%d  depth=%d  | ESC to quit\n",
            scene_name.c_str(), W, H, spp, depth);

    // --- Render loop ---
    auto t0     = std::chrono::steady_clock::now();
    auto t_start = t0;
    int  frames = 0;

    // Base vup and forward axis for roll
    vec3  base_vup   = cam.vup;
    vec3  fwd        = unit_vector(cam.lookat - cam.lookfrom);
    float roll_speed = 0.3f;  // radians per second

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        // Roll: rotate vup around the forward axis
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        float a   = (float)elapsed * roll_speed;
        float c   = cosf(a), sr = sinf(a);
        // Rodrigues' rotation: v*cos + cross(fwd,v)*sin + fwd*dot(fwd,v)*(1-cos)
        cam.vup  = base_vup * c
                 + cross(fwd, base_vup) * sr
                 + fwd * dot(fwd, base_vup) * (1.f - c);
        cam.initialize();

        // Render on GPU → d_rgba (gamma-corrected uchar4)
        cuda_render_frame_rt(rs, cam, scene);

        // Copy to host and upload to GL texture
        cudaMemcpy(h_rgba.data(), rs.d_rgba,
                   W * H * sizeof(uchar4), cudaMemcpyDeviceToHost);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RGBA, GL_UNSIGNED_BYTE, h_rgba.data());

        // Draw fullscreen quad.
        // Our framebuffer has row 0 at top; GL texture has (0,0) at bottom-left,
        // so we flip the T coordinate.
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
          glTexCoord2f(0.f, 0.f); glVertex2f(0.f, 1.f);  // screen top-left
          glTexCoord2f(1.f, 0.f); glVertex2f(1.f, 1.f);  // screen top-right
          glTexCoord2f(1.f, 1.f); glVertex2f(1.f, 0.f);  // screen bottom-right
          glTexCoord2f(0.f, 1.f); glVertex2f(0.f, 0.f);  // screen bottom-left
        glEnd();

        glfwSwapBuffers(window);

        // FPS display in title bar (updated every 0.5 s)
        ++frames;
        auto t1  = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s >= 0.5) {
            char title[128];
            snprintf(title, sizeof(title),
                     "Ray Tracer  |  %.1f FPS  |  spp=%d  depth=%d  |  %dx%d",
                     frames / s, spp, depth, W, H);
            glfwSetWindowTitle(window, title);
            frames = 0;
            t0     = t1;
        }
    }

    // --- Cleanup ---
    cuda_render_cleanup(rs);
    free_device_scene(scene);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
