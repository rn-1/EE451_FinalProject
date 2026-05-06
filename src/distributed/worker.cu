// Distributed ray tracer — Worker node (mapper).
// Runs on the remote GPU machine (RTX 3070 Ti).
// Listens on TCP, renders assigned horizontal tile per frame, sends pixels
// back.
//
// Usage: ./bin/worker_rt [--scene random|cornell] [--port 9000] [--device 0]

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../include/utils.h"
#include "../cuda/device_scene.cuh"
#include "../cuda/render.cuh"
#include "protocol.h"

int main(int argc, char **argv) {
  std::string scene_name = get_arg(argc, argv, "--scene", "random");
  int port = std::stoi(get_arg(argc, argv, "--port", "9000"));
  int device_id = std::stoi(get_arg(argc, argv, "--device", "0"));

  cudaSetDevice(device_id);

  // --- Build scene once (same deterministic seed as master) ---
  DeviceScene scene;
  if (scene_name == "cornell")
    scene = build_device_cornell_box();
  else if (scene_name == "complex")
    scene = build_device_complex_scene();
  else
    scene = build_device_random_spheres();

  fprintf(stderr, "Worker | scene=%s | GPU device=%d\n", scene_name.c_str(),
          device_id);

  // --- TCP server setup ---
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    perror("socket");
    return 1;
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // TCP_NODELAY: send pixel data immediately without Nagle buffering
  {
    int f = 1;
    setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f));
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);

  if (bind(server_fd, (sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind");
    return 1;
  }
  if (listen(server_fd, 1) < 0) {
    perror("listen");
    return 1;
  }

  fprintf(stderr, "Worker | listening on port %d | waiting for master...\n",
          port);

  // --- Accept one persistent connection from master ---
  sockaddr_in client_addr{};
  socklen_t client_len = sizeof(client_addr);
  int conn = accept(server_fd, (sockaddr *)&client_addr, &client_len);
  if (conn < 0) {
    perror("accept");
    return 1;
  }

  char ip_str[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, sizeof(ip_str));
  fprintf(stderr, "Worker | master connected from %s\n", ip_str);

  // --- Receive first request to learn tile dimensions ---
  RenderRequest req{};
  if (!recv_all(conn, &req, sizeof(req))) {
    fprintf(stderr, "Worker | failed to receive first request\n");
    return 1;
  }

  int tile_h = req.row_end - req.row_start;
  int tile_w = req.width;

  fprintf(stderr, "Worker | tile %dx%d (rows %d..%d) spp=%d depth=%d\n", tile_w,
          tile_h, req.row_start, req.row_end, req.spp, req.depth);

  // --- Allocate persistent tile GPU state ---
  TileRenderState ts;
  cuda_tile_render_init(ts, tile_w, tile_h);

  std::vector<uchar4> h_rgba(tile_w * tile_h);

  // --- Process first request already received ---
  bool first = true;
  while (true) {
    if (!first) {
      if (!recv_all(conn, &req, sizeof(req))) {
        fprintf(stderr, "Worker | master disconnected\n");
        break;
      }
    }
    first = false;

    // Render tile on GPU
    cuda_tile_render_frame(ts, req.cam, scene, req.row_start);

    // Copy result to host
    cudaMemcpy(h_rgba.data(), ts.d_rgba, tile_w * tile_h * sizeof(uchar4),
               cudaMemcpyDeviceToHost);

    // Send response header + pixel data
    RenderResponse resp{};
    resp.row_start = req.row_start;
    resp.row_end = req.row_end;
    resp.num_pixels = tile_w * tile_h;

    if (!send_all(conn, &resp, sizeof(resp)))
      break;
    if (!send_all(conn, h_rgba.data(), resp.num_pixels * sizeof(uchar4)))
      break;
  }

  cuda_tile_render_cleanup(ts);
  free_device_scene(scene);
  close(conn);
  close(server_fd);
  return 0;
}
