// CUDA render kernels and host orchestration.

#include "render.cuh"
#include "device_scene.cuh"
#include "../../include/utils.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>

#define TILE_W 16
#define TILE_H 16

// ============================================================
//  Device material scatter (switch on MatType, no vtable)
// ============================================================

__device__ static float schlick(float cosine, float ref_idx) {
    float r0 = (1.f - ref_idx) / (1.f + ref_idx);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(1.f - cosine, 5.f);
}

__device__ static bool device_scatter(const MaterialData& mat,
                                       const ray& r_in,
                                       const CudaHitRecord& rec,
                                       color& attenuation,
                                       ray& scattered,
                                       curandState* rs) {
    switch (mat.type) {
    case MatType::Lambertian: {
        vec3 dir = rec.normal + random_unit_vector(rs);
        if (dir.near_zero()) dir = rec.normal;
        scattered   = ray(rec.p, dir);
        attenuation = mat.albedo;
        return true;
    }
    case MatType::Metal: {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        reflected = unit_vector(reflected) + mat.fuzz * random_unit_vector(rs);
        scattered   = ray(rec.p, reflected);
        attenuation = mat.albedo;
        return dot(scattered.direction(), rec.normal) > 0.f;
    }
    case MatType::Dielectric: {
        attenuation = color(1.f, 1.f, 1.f);
        float ratio = rec.front_face ? (1.f / mat.ir) : mat.ir;
        vec3  ud    = unit_vector(r_in.direction());
        float cos_t = fminf(dot(-ud, rec.normal), 1.f);
        float sin_t = sqrtf(1.f - cos_t * cos_t);
        bool cannot_refract = ratio * sin_t > 1.f;
        vec3 dir;
        if (cannot_refract || schlick(cos_t, ratio) > rand_float(rs))
            dir = reflect(ud, rec.normal);
        else
            dir = refract(ud, rec.normal, ratio);
        scattered = ray(rec.p, dir);
        return true;
    }
    case MatType::DiffuseLight:
        return false;  // lights don't scatter
    }
    return false;
}

// ============================================================
//  Iterative ray_color — no recursion, no growing device stack
// ============================================================

__device__ static color ray_color_iterative(ray r, int max_depth,
                                             const DeviceScene& scene,
                                             curandState* rs,
                                             unsigned long long& ray_count) {
    color accumulated_color(0.f, 0.f, 0.f);
    color running_atten(1.f, 1.f, 1.f);

    for (int depth = 0; depth < max_depth; ++depth) {
        ++ray_count;
        // Find nearest hit
        CudaHitRecord rec;
        bool hit_anything = false;
        float closest = 1e30f;

        for (int i = 0; i < scene.num_hittables; ++i) {
            CudaHitRecord tmp;
            if (scene.d_hittables[i]->hit(r, 0.001f, closest, tmp)) {
                hit_anything = true;
                closest = tmp.t;
                rec = tmp;
            }
        }

        if (!hit_anything) {
            // Sky or black background
            if (scene.background.x() == 0.f &&
                scene.background.y() == 0.f &&
                scene.background.z() == 0.f) {
                // No sky — black (Cornell Box mode)
            } else {
                // Sky gradient
                vec3 unit = unit_vector(r.direction());
                float t = 0.5f * (unit.y() + 1.f);
                color sky = (1.f - t) * color(1.f, 1.f, 1.f) + t * scene.background;
                accumulated_color += running_atten * sky;
            }
            break;
        }

        const MaterialData& mat = scene.d_materials[rec.mat_id];

        // Emissive
        if (mat.type == MatType::DiffuseLight) {
            accumulated_color += running_atten * mat.emit_color;
            break;
        }

        // Scatter
        color attenuation;
        ray   scattered;
        if (!device_scatter(mat, r, rec, attenuation, scattered, rs))
            break;

        running_atten *= attenuation;
        r = scattered;
    }

    return accumulated_color;
}

// ============================================================
//  Kernel 1: Initialize cuRAND states
// ============================================================

__global__ void kernel_init_curand(curandState* states,
                                    int width, int height,
                                    unsigned long long seed) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    // Each thread: unique sequence, no offset
    curand_init(seed, (unsigned long long)idx, 0, &states[idx]);
}

// ============================================================
//  Kernel 2: Main render kernel — one thread per pixel
// ============================================================

__global__ void kernel_render(curandState* states,
                               CameraParams cam,
                               DeviceScene  scene,
                               color*       fb,
                               int width, int height,
                               unsigned long long* d_ray_count) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int pixel_idx = row * width + col;

    // Load cuRAND state to registers for this thread
    curandState local_state = states[pixel_idx];
    unsigned long long local_ray_count = 0ULL;

    color pixel_color(0.f, 0.f, 0.f);

    for (int s = 0; s < cam.samples_per_pixel; ++s) {
        // Stratified sample: random sub-pixel offset
        float offset_u = curand_uniform(&local_state) - 0.5f;
        float offset_v = curand_uniform(&local_state) - 0.5f;

        point3 pixel_center = cam.pixel00_loc
                            + (col + offset_u) * cam.pixel_delta_u
                            + (row + offset_v) * cam.pixel_delta_v;

        point3 ray_origin;
        if (cam.defocus_angle <= 0.f) {
            ray_origin = cam.center;
        } else {
            vec3 p = random_in_unit_disk(&local_state);
            ray_origin = cam.center
                       + p.x() * cam.defocus_disk_u
                       + p.y() * cam.defocus_disk_v;
        }

        ray r(ray_origin, pixel_center - ray_origin);
        pixel_color += ray_color_iterative(r, cam.max_depth, scene, &local_state, local_ray_count);
    }

    // Write back cuRAND state and averaged color
    states[pixel_idx] = local_state;
    fb[pixel_idx] = pixel_color / (float)cam.samples_per_pixel;

    if (local_ray_count > 0) {
        atomicAdd(d_ray_count, local_ray_count);
    }
}

// ============================================================
//  Host orchestration: cuda_render()
// ============================================================

void cuda_render(const CameraParams& cam,
                 const DeviceScene&  scene,
                 color*              h_fb,
                 float*              kernel_ms_out,
                 double*             total_ms_out,
                 unsigned long long* total_ray_count_out) {

    int W = cam.image_width;
    int H = cam.image_height;
    int N = W * H;

    Timer wall_timer;
    wall_timer.begin();

    // Allocate device framebuffer
    color* d_fb = nullptr;
    cudaMalloc(&d_fb, N * sizeof(color));

    // Allocate cuRAND states
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, N * sizeof(curandState));

    // Grid/block layout
    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    // Initialize cuRAND (separate from render time)
    kernel_init_curand<<<grid, block>>>(d_states, W, H, 12345ULL);
    cudaDeviceSynchronize();

    unsigned long long* d_ray_count = nullptr;
    cudaMalloc(&d_ray_count, sizeof(unsigned long long));
    cudaMemset(d_ray_count, 0, sizeof(unsigned long long));

    // --- Timed render kernel ---
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    kernel_render<<<grid, block>>>(d_states, cam, scene, d_fb, W, H, d_ray_count);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Copy result back to host
    unsigned long long total_ray_count = 0ULL;
    cudaMemcpy(h_fb, d_fb, N * sizeof(color), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_ray_count, d_ray_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double total_ms = wall_timer.elapsed_ms();

    cudaFree(d_fb);
    cudaFree(d_states);
    cudaFree(d_ray_count);

    if (kernel_ms_out) *kernel_ms_out = kernel_ms;
    if (total_ms_out)  *total_ms_out  = total_ms;
    if (total_ray_count_out) *total_ray_count_out = total_ray_count;
}
