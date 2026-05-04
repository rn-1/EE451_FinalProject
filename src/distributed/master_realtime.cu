// Distributed ray tracer — Master node (reducer).
// Runs locally (RTX 3060).  Splits frame into top/bottom halves:
//   top  (rows 0..H/2)  → rendered locally on RTX 3060
//   bottom (rows H/2..H) → sent to remote worker (RTX 3070 Ti) via TCP
// Both halves are computed in parallel; master combines and displays via GLFW.
//
// Usage:
//   ./bin/master_realtime_rt --worker-ip <IP> [--worker-port 9000]
//                            [--scene random|cornell] [--width 1280]
//                            [--spp 4] [--depth 4] [--device 0]

#include <GLFW/glfw3.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../cuda/render.cuh"
#include "../cuda/device_scene.cuh"
#include "../../include/utils.h"
#include "../../include/camera.h"
#include "protocol.h"

int main(int argc, char** argv) {
    std::string worker_ip  = get_arg(argc, argv, "--worker-ip",   "127.0.0.1");
    int worker_port = std::stoi(get_arg(argc, argv, "--worker-port", "9000"));
    std::string scene_name = get_arg(argc, argv, "--scene",        "random");
    int img_width  = std::stoi(get_arg(argc, argv, "--width",      "1280"));
    int spp        = std::stoi(get_arg(argc, argv, "--spp",        "4"));
    int depth      = std::stoi(get_arg(argc, argv, "--depth",      "4"));
    int device_id  = std::stoi(get_arg(argc, argv, "--device",     "0"));

    cudaSetDevice(device_id);

    // --- Camera setup ---
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

    // Horizontal split: local = top half, remote = bottom half.
    // If H is odd, give the extra row to the local node.
    int local_row_start  = 0;
    int local_row_end    = H / 2 + (H % 2);  // top
    int remote_row_start = local_row_end;
    int remote_row_end   = H;                 // bottom

    int local_tile_h  = local_row_end  - local_row_start;
    int remote_tile_h = remote_row_end - remote_row_start;

    // --- Connect to worker ---
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    // TCP_NODELAY: send RenderRequest immediately without Nagle buffering
    { int f = 1; setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f)); }

    // 5-second timeout on recv so a slow/dropped connection doesn't freeze the loop
    { struct timeval tv{5, 0};
      setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)); }

    sockaddr_in waddr{};
    waddr.sin_family = AF_INET;
    waddr.sin_port   = htons((uint16_t)worker_port);
    if (inet_pton(AF_INET, worker_ip.c_str(), &waddr.sin_addr) <= 0) {
        fprintf(stderr, "Master | invalid worker IP: %s\n", worker_ip.c_str());
        return 1;
    }
    if (connect(sock, (sockaddr*)&waddr, sizeof(waddr)) < 0) {
        perror("connect");
        fprintf(stderr, "Master | could not connect to worker at %s:%d\n",
                worker_ip.c_str(), worker_port);
        return 1;
    }
    fprintf(stderr, "Master | connected to worker %s:%d\n",
            worker_ip.c_str(), worker_port);

    // --- Local scene + tile render state ---
    DeviceScene scene;
    if (scene_name == "cornell")
        scene = build_device_cornell_box();
    else
        scene = build_device_random_spheres();

    TileRenderState local_ts;
    cuda_tile_render_init(local_ts, W, local_tile_h);

    // --- GLFW window ---
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(W, H, "Ray Tracer (Distributed)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,1,0,1,-1,1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();

    // Full-frame host buffer (combined top + bottom)
    std::vector<uchar4> h_full(W * H);
    std::vector<uchar4> h_local(W * local_tile_h);
    std::vector<uchar4> h_remote(W * remote_tile_h);

    // Camera roll state
    vec3  base_vup   = cam.vup;
    vec3  fwd        = unit_vector(cam.lookat - cam.lookfrom);
    float roll_speed = 0.3f;

    fprintf(stderr,
            "Master | scene=%s  res=%dx%d  spp=%d  depth=%d\n"
            "       | local  rows %d..%d (%d rows) → RTX 3060\n"
            "       | remote rows %d..%d (%d rows) → RTX 3070 Ti\n"
            "       | ESC to quit\n",
            scene_name.c_str(), W, H, spp, depth,
            local_row_start,  local_row_end,  local_tile_h,
            remote_row_start, remote_row_end, remote_tile_h);

    auto t0      = std::chrono::steady_clock::now();
    auto t_start = t0;
    int  frames  = 0;
    bool worker_ok = true;

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

        // ── MAPPER 2: network thread sends request to worker, receives bottom half ──
        bool net_ok = false;
        std::thread net_thread([&]() {
            if (!worker_ok) return;

            RenderRequest req{};
            req.width      = W;
            req.height     = H;
            req.row_start  = remote_row_start;
            req.row_end    = remote_row_end;
            req.spp        = spp;
            req.depth      = depth;
            strncpy(req.scene_name, scene_name.c_str(), 31);
            req.cam        = cam;

            if (!send_all(sock, &req, sizeof(req))) return;

            RenderResponse resp{};
            if (!recv_all(sock, &resp, sizeof(resp))) return;
            if (!recv_all(sock, h_remote.data(),
                          resp.num_pixels * sizeof(uchar4))) return;
            net_ok = true;
        });

        // ── MAPPER 1: local GPU renders top half ──
        cuda_tile_render_frame(local_ts, cam, scene, local_row_start);
        cudaMemcpy(h_local.data(), local_ts.d_rgba,
                   W * local_tile_h * sizeof(uchar4),
                   cudaMemcpyDeviceToHost);

        net_thread.join();
        if (!net_ok) {
            fprintf(stderr, "Master | worker disconnected, continuing local-only\n");
            worker_ok = false;
        }

        // ── REDUCER: combine top + bottom into full framebuffer ──
        memcpy(h_full.data(),
               h_local.data(),
               W * local_tile_h * sizeof(uchar4));

        if (worker_ok) {
            memcpy(h_full.data() + W * local_tile_h,
                   h_remote.data(),
                   W * remote_tile_h * sizeof(uchar4));
        }
        // (if worker failed, bottom half shows whatever was last received)

        // Upload full frame to GL texture and draw
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RGBA, GL_UNSIGNED_BYTE, h_full.data());

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
            char title[160];
            snprintf(title, sizeof(title),
                     "Ray Tracer (Distributed%s)  |  %.1f FPS  |  spp=%d depth=%d  |  %dx%d",
                     worker_ok ? "" : " — worker offline",
                     frames / s, spp, depth, W, H);
            glfwSetWindowTitle(window, title);
            frames = 0; t0 = t1;
        }
    }

    cuda_tile_render_cleanup(local_ts);
    free_device_scene(scene);
    close(sock);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
