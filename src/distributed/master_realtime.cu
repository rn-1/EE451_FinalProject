// Distributed ray tracer — Master node (reducer), multi-worker.
// Splits frame into (1 + N_workers) equal horizontal tiles.
// Tile 0: local GPU.  Tiles 1..N: sent to remote workers via TCP in parallel.
//
// Usage (single worker, backward compat):
//   ./bin/master_realtime_rt --worker-ip <IP> [--worker-port 9000] ...
//
// Usage (multiple workers — set NUM_NODES here):
//   ./bin/master_realtime_rt \
//       --workers "IP1:PORT1,IP2:PORT2,IP3:PORT3" \
//       [--scene random|cornell] [--width 1280] [--spp 4] [--depth 4]
//
// Number of rendering nodes = 1 (local GPU) + number of --workers entries.

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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../../include/camera.h"
#include "../../include/utils.h"
#include "../cuda/device_scene.cuh"
#include "../cuda/render.cuh"
#include "protocol.h"

// ── Per-worker connection state ──────────────────────────────────────────────

struct WorkerConn {
  std::string ip;
  int port = 9000;
  int sock = -1;
  int row_start = 0;
  int row_end = 0;
  bool ok = true;
};

static int connect_worker(const std::string &ip, int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0)
    return -1;
  {
    int f = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f));
  }
  {
    struct timeval tv{5, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    close(s);
    return -1;
  }
  if (connect(s, (sockaddr *)&addr, sizeof(addr)) < 0) {
    close(s);
    return -1;
  }
  return s;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {

  // ── Build worker list ────────────────────────────────────────────────────
  // Option A (multi): --workers "IP1:PORT1,IP2:PORT2,..."
  // Option B (legacy single): --worker-ip IP [--worker-port 9000]
  std::vector<WorkerConn> workers;

  std::string workers_arg = get_arg(argc, argv, "--workers", "");
  if (!workers_arg.empty()) {
    std::istringstream ss(workers_arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
      WorkerConn w;
      auto colon = token.rfind(':');
      if (colon != std::string::npos) {
        w.ip = token.substr(0, colon);
        w.port = std::stoi(token.substr(colon + 1));
      } else {
        w.ip = token;
        w.port = 9000;
      }
      workers.push_back(w);
    }
  } else {
    std::string ip = get_arg(argc, argv, "--worker-ip", "");
    if (!ip.empty()) {
      WorkerConn w;
      w.ip = ip;
      w.port = std::stoi(get_arg(argc, argv, "--worker-port", "9000"));
      workers.push_back(w);
    }
  }

  std::string scene_name = get_arg(argc, argv, "--scene", "random");
  int img_width = std::stoi(get_arg(argc, argv, "--width", "1280"));
  int spp = std::stoi(get_arg(argc, argv, "--spp", "4"));
  int depth = std::stoi(get_arg(argc, argv, "--depth", "4"));
  int device_id = std::stoi(get_arg(argc, argv, "--device", "0"));

  cudaSetDevice(device_id);

  // Total rendering nodes: 1 local + N remote
  int num_nodes = 1 + (int)workers.size();

  // ── Camera ───────────────────────────────────────────────────────────────
  CameraParams cam;
  if (scene_name == "cornell") {
    cam.aspect_ratio = 1.f;
    cam.vfov = 40.f;
    cam.lookfrom = point3(278.f, 278.f, -800.f);
    cam.lookat = point3(278.f, 278.f, 0.f);
    cam.vup = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle = 0.f;
    cam.focus_dist = 10.f;
    cam.background = color(0.f, 0.f, 0.f);
  } else {
    cam.aspect_ratio = 16.f / 9.f;
    cam.vfov = 20.f;
    cam.lookfrom = point3(13.f, 2.f, 3.f);
    cam.lookat = point3(0.f, 0.f, 0.f);
    cam.vup = vec3(0.f, 1.f, 0.f);
    cam.defocus_angle = 0.6f;
    cam.focus_dist = 10.f;
    cam.background = color(0.5f, 0.7f, 1.f);
  }
  cam.image_width = img_width;
  cam.samples_per_pixel = spp;
  cam.max_depth = depth;
  cam.initialize();

  int W = cam.image_width;
  int H = cam.image_height;

  // ── Tile boundaries ───────────────────────────────────────────────────────
  // Divide H into num_nodes equal strips; last strip absorbs any remainder.
  int base_h = H / num_nodes;

  int local_row_start = 0;
  int local_row_end = (num_nodes == 1) ? H : base_h; // all rows if solo
  int local_tile_h = local_row_end - local_row_start;

  for (int i = 0; i < (int)workers.size(); ++i) {
    workers[i].row_start = base_h * (i + 1);
    workers[i].row_end = (i == (int)workers.size() - 1) ? H : base_h * (i + 2);
  }

  // ── Connect to workers ───────────────────────────────────────────────────
  for (auto &w : workers) {
    w.sock = connect_worker(w.ip, w.port);
    if (w.sock < 0) {
      fprintf(stderr,
              "Master | WARN: could not connect to worker %s:%d — skipping\n",
              w.ip.c_str(), w.port);
      w.ok = false;
    } else {
      fprintf(stderr,
              "Master | connected to worker %s:%d  (rows %d..%d, %d rows)\n",
              w.ip.c_str(), w.port, w.row_start, w.row_end,
              w.row_end - w.row_start);
    }
  }

  // ── Local scene + tile state ─────────────────────────────────────────────
  DeviceScene scene;
  if (scene_name == "cornell")
    scene = build_device_cornell_box();
  else if (scene_name == "complex")
    scene = build_device_complex_scene();
  else
    scene = build_device_random_spheres();

  TileRenderState local_ts;
  cuda_tile_render_init(local_ts, W, local_tile_h);

  // ── GLFW window + GL texture ─────────────────────────────────────────────
  if (!glfwInit()) {
    fprintf(stderr, "GLFW init failed\n");
    return 1;
  }
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow *window =
      glfwCreateWindow(W, H, "Ray Tracer (Distributed)", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // ── Host pixel buffers ────────────────────────────────────────────────────
  std::vector<uchar4> h_full(W * H);
  std::vector<uchar4> h_local(W * local_tile_h);

  std::vector<std::vector<uchar4>> h_remote(workers.size());
  for (int i = 0; i < (int)workers.size(); ++i)
    h_remote[i].resize(W * (workers[i].row_end - workers[i].row_start));

  // ── Camera roll state ─────────────────────────────────────────────────────
  vec3 base_vup = cam.vup;
  vec3 fwd = unit_vector(cam.lookat - cam.lookfrom);
  float roll_speed = 0.3f;

  // ── Print summary ─────────────────────────────────────────────────────────
  fprintf(stderr,
          "Master | scene=%s  res=%dx%d  spp=%d  depth=%d  nodes=%d\n"
          "       | local rows %d..%d (%d rows) → local GPU\n",
          scene_name.c_str(), W, H, spp, depth, num_nodes, local_row_start,
          local_row_end, local_tile_h);
  for (int i = 0; i < (int)workers.size(); ++i)
    fprintf(stderr, "       | worker[%d] %s:%d rows %d..%d (%d rows)\n", i,
            workers[i].ip.c_str(), workers[i].port, workers[i].row_start,
            workers[i].row_end, workers[i].row_end - workers[i].row_start);
  fprintf(stderr, "       | ESC to quit\n");

  auto t0 = std::chrono::steady_clock::now();
  auto t_start = t0;
  int frames = 0;

  // ── Render loop ───────────────────────────────────────────────────────────
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GLFW_TRUE);

    // Camera roll (Rodrigues around forward axis)
    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_start)
                         .count();
    float a = (float)elapsed * roll_speed;
    float c = cosf(a), sr = sinf(a);
    cam.vup = base_vup * c + cross(fwd, base_vup) * sr +
              fwd * dot(fwd, base_vup) * (1.f - c);
    cam.initialize();

    // ── MAPPERS: one thread per worker, run in parallel with local GPU ────
    std::vector<bool> net_ok(workers.size(), false);
    std::vector<std::thread> net_threads;
    net_threads.reserve(workers.size());

    for (int i = 0; i < (int)workers.size(); ++i) {
      net_threads.emplace_back([&, i]() {
        WorkerConn &w = workers[i];
        if (!w.ok)
          return;

        RenderRequest req{};
        req.width = W;
        req.height = H;
        req.row_start = w.row_start;
        req.row_end = w.row_end;
        req.spp = spp;
        req.depth = depth;
        strncpy(req.scene_name, scene_name.c_str(), 31);
        req.cam = cam;

        if (!send_all(w.sock, &req, sizeof(req)))
          return;

        RenderResponse resp{};
        if (!recv_all(w.sock, &resp, sizeof(resp)))
          return;
        if (!recv_all(w.sock, h_remote[i].data(),
                      resp.num_pixels * sizeof(uchar4)))
          return;
        net_ok[i] = true;
      });
    }

    // ── MAPPER 0: local GPU renders its tile ──────────────────────────────
    cuda_tile_render_frame(local_ts, cam, scene, local_row_start);
    cudaMemcpy(h_local.data(), local_ts.d_rgba,
               W * local_tile_h * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // Wait for all network threads
    for (auto &t : net_threads)
      t.join();

    for (int i = 0; i < (int)workers.size(); ++i) {
      if (!net_ok[i] && workers[i].ok) {
        fprintf(stderr, "Master | worker[%d] %s:%d disconnected\n", i,
                workers[i].ip.c_str(), workers[i].port);
        workers[i].ok = false;
      }
    }

    // ── REDUCER: assemble full frame from all tiles ───────────────────────
    memcpy(h_full.data(), h_local.data(), W * local_tile_h * sizeof(uchar4));

    for (int i = 0; i < (int)workers.size(); ++i) {
      if (!workers[i].ok)
        continue; // keep stale data visible
      int th = workers[i].row_end - workers[i].row_start;
      memcpy(h_full.data() + W * workers[i].row_start, h_remote[i].data(),
             W * th * sizeof(uchar4));
    }

    // Upload full frame + draw
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE,
                    h_full.data());

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.f);
    glVertex2f(0.f, 1.f);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(1.f, 1.f);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(1.f, 0.f);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(0.f, 0.f);
    glEnd();

    glfwSwapBuffers(window);

    ++frames;
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    if (s >= 0.5) {
      char title[256];
      snprintf(title, sizeof(title),
               "Ray Tracer (Distributed, %d nodes)  |  %.1f FPS  |  spp=%d "
               "depth=%d  |  %dx%d",
               num_nodes, frames / s, spp, depth, W, H);
      glfwSetWindowTitle(window, title);
      frames = 0;
      t0 = t1;
    }
  }

  cuda_tile_render_cleanup(local_ts);
  free_device_scene(scene);
  for (auto &w : workers)
    if (w.sock >= 0)
      close(w.sock);
  glDeleteTextures(1, &tex);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
