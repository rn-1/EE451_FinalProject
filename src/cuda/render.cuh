#pragma once

#include "../../include/vec3.h"
#include "../../include/camera.h"

// Forward declaration of device scene descriptor
struct DeviceScene;

// Host-callable render function.
// Allocates device framebuffer, launches kernels, copies result back to h_fb.
// h_fb must be pre-allocated: cam.image_width * cam.image_height * sizeof(color)
// kernel_ms_out: GPU-side time (render kernel only, cudaEvent)
// total_ms_out:  wall-clock including H2D, cuRAND init, D2H
void cuda_render(const CameraParams& cam,
                 const DeviceScene&  scene,
                 color*              h_fb,
                 float*              kernel_ms_out,
                 double*             total_ms_out,
                 unsigned long long* total_ray_count_out = nullptr);
