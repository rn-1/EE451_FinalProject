#pragma once

#include "../../include/vec3.h"
#include "../../include/camera.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

struct DeviceScene;

void cuda_render(const CameraParams& cam,
                 const DeviceScene&  scene,
                 color*              h_fb,
                 float*              kernel_ms_out,
                 double*             total_ms_out,
                 unsigned long long* total_ray_count_out = nullptr);

// Persistent GPU state for the real-time render loop.
// Allocate once with cuda_render_init; reuse every frame.
struct RenderState {
    curandState* d_states = nullptr;
    color*       d_fb     = nullptr;
    uchar4*      d_rgba   = nullptr;
    int          width    = 0;
    int          height   = 0;
};

void cuda_render_init(RenderState& rs, int width, int height);
void cuda_render_frame_rt(RenderState& rs, const CameraParams& cam, const DeviceScene& scene);
void cuda_render_cleanup(RenderState& rs);

// Tile render state — like RenderState but sized for a horizontal strip.
// row_offset passed per-frame so the kernel computes correct world-space rays.
struct TileRenderState {
    curandState* d_states    = nullptr;
    color*       d_fb        = nullptr;
    uchar4*      d_rgba      = nullptr;
    int          tile_width  = 0;
    int          tile_height = 0;
};

void cuda_tile_render_init(TileRenderState& ts, int tile_w, int tile_h);
void cuda_tile_render_frame(TileRenderState& ts, const CameraParams& cam,
                             const DeviceScene& scene, int row_offset);
void cuda_tile_render_cleanup(TileRenderState& ts);
