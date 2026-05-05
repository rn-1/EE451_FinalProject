#pragma once

#include "../../include/camera.h"

static constexpr int WORKER_PORT = 9000;

// Sent master → worker every frame (~200 bytes total).
// Camera is fully initialized (pixel00_loc, pixel_delta_u/v already computed),
// so the worker does not need to call cam.initialize().
struct RenderRequest {
    int          width, height;   // full image dimensions
    int          row_start;       // first row this worker renders (inclusive)
    int          row_end;         // last row this worker renders (exclusive)
    int          spp, depth;
    char         scene_name[32];  // "random" or "cornell"
    CameraParams cam;
};

// Sent worker → master as a header; followed immediately by
// num_pixels * sizeof(uchar4) bytes of gamma-corrected RGBA pixel data.
struct RenderResponse {
    int row_start, row_end;
    int num_pixels;   // width * (row_end - row_start)
};

// Helpers: send/recv exact byte counts over a blocking socket.
// Returns true on success, false on error/disconnect.
#include <sys/socket.h>
#include <cstddef>

inline bool send_all(int fd, const void* buf, size_t len) {
    const char* p = reinterpret_cast<const char*>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= (size_t)n;
    }
    return true;
}

inline bool recv_all(int fd, void* buf, size_t len) {
    char* p = reinterpret_cast<char*>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return false;
        p   += n;
        len -= (size_t)n;
    }
    return true;
}
