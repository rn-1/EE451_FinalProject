// Device-side scene construction using device-new inside <<<1,1>>> kernels.
// This is required so that virtual function tables are in device memory.

#include "device_scene.cuh"
#include <curand_kernel.h>
#include <cstdio>

// ---- Simple LCG for deterministic device-side scene generation ----
__device__ static unsigned int lcg_state = 1337u;
__device__ float lcg_rand() {
    lcg_state = 1664525u * lcg_state + 1013904223u;
    return (float)(lcg_state & 0x00FFFFFFu) / (float)0x01000000u;
}

// ============================================================
//  Random sphere scene builder
// ============================================================

__global__ void kernel_build_random_spheres(DeviceHittable** list,
                                             MaterialData*    mats,
                                             int*             num_hittables_out,
                                             int*             num_mats_out) {
    // Single-thread kernel (called with <<<1,1>>>)
    int h_idx = 0;
    int m_idx = 0;

    // Ground sphere
    mats[m_idx] = {MatType::Lambertian, color(0.5f, 0.5f, 0.5f), 0.f, 0.f, color(0,0,0)};
    list[h_idx++] = new DeviceSphere(point3(0.f, -1000.f, 0.f), 1000.f, m_idx++);

    // Random small spheres (deterministic via LCG)
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose = lcg_rand();
            point3 center(a + 0.9f * lcg_rand(), 0.2f, b + 0.9f * lcg_rand());

            // Skip if too close to big feature spheres
            vec3 diff = center - point3(4.f, 0.2f, 0.f);
            if (dot(diff, diff) <= 0.9f * 0.9f) continue;

            if (choose < 0.8f) {
                // Lambertian
                color alb(lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand());
                mats[m_idx] = {MatType::Lambertian, alb, 0.f, 0.f, color(0,0,0)};
            } else if (choose < 0.95f) {
                // Metal
                color alb(0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand());
                float fuzz = 0.5f * lcg_rand();
                mats[m_idx] = {MatType::Metal, alb, fuzz, 0.f, color(0,0,0)};
            } else {
                // Dielectric
                mats[m_idx] = {MatType::Dielectric, color(1,1,1), 0.f, 1.5f, color(0,0,0)};
            }
            list[h_idx++] = new DeviceSphere(center, 0.2f, m_idx++);
        }
    }

    // Three large feature spheres
    mats[m_idx] = {MatType::Dielectric, color(1,1,1), 0.f, 1.5f, color(0,0,0)};
    list[h_idx++] = new DeviceSphere(point3(0.f, 1.f, 0.f), 1.f, m_idx++);

    mats[m_idx] = {MatType::Lambertian, color(0.4f, 0.2f, 0.1f), 0.f, 0.f, color(0,0,0)};
    list[h_idx++] = new DeviceSphere(point3(-4.f, 1.f, 0.f), 1.f, m_idx++);

    mats[m_idx] = {MatType::Metal, color(0.7f, 0.6f, 0.5f), 0.f, 0.f, color(0,0,0)};
    list[h_idx++] = new DeviceSphere(point3(4.f, 1.f, 0.f), 1.f, m_idx++);

    *num_hittables_out = h_idx;
    *num_mats_out      = m_idx;
}

// ============================================================
//  Cornell Box scene builder
// ============================================================

__global__ void kernel_build_cornell_box(DeviceHittable** list,
                                          MaterialData*    mats,
                                          int*             num_hittables_out,
                                          int*             num_mats_out) {
    int h_idx = 0;
    int m_idx = 0;

    // Materials
    int red_id   = m_idx; mats[m_idx++] = {MatType::Lambertian, color(0.65f,0.05f,0.05f), 0,0,color(0,0,0)};
    int white_id = m_idx; mats[m_idx++] = {MatType::Lambertian, color(0.73f,0.73f,0.73f), 0,0,color(0,0,0)};
    int green_id = m_idx; mats[m_idx++] = {MatType::Lambertian, color(0.12f,0.45f,0.15f), 0,0,color(0,0,0)};
    int light_id = m_idx; mats[m_idx++] = {MatType::DiffuseLight, color(0,0,0), 0,0,color(15.f,15.f,15.f)};
    int metal_id = m_idx; mats[m_idx++] = {MatType::Metal, color(0.75f,0.75f,0.75f), 0, 0, color(0,0,0)};

    // Walls
    list[h_idx++] = new DeviceQuad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), red_id);    // left
    list[h_idx++] = new DeviceQuad(point3(0,0,0),   vec3(0,555,0), vec3(0,0,555), green_id);  // right
    list[h_idx++] = new DeviceQuad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light_id); // ceiling light
    list[h_idx++] = new DeviceQuad(point3(0,0,0),   vec3(555,0,0), vec3(0,0,555), white_id);  // floor
    list[h_idx++] = new DeviceQuad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white_id); // ceiling
    list[h_idx++] = new DeviceQuad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), metal_id);  // back wall

    // Helper lambda can't be used in device code, so inline the 6-quad box:
    // Short box: [130,0,65] - [295,165,230]
    {
        float x0=130,y0=0,z0=65, x1=295,y1=165,z1=230;
        list[h_idx++] = new DeviceQuad(point3(x0,y0,z1), vec3(x1-x0,0,0), vec3(0,y1-y0,0), white_id); // front
        list[h_idx++] = new DeviceQuad(point3(x1,y0,z0), vec3(x0-x1,0,0), vec3(0,y1-y0,0), white_id); // back
        list[h_idx++] = new DeviceQuad(point3(x0,y0,z0), vec3(0,0,z1-z0), vec3(0,y1-y0,0), white_id); // left
        list[h_idx++] = new DeviceQuad(point3(x1,y0,z1), vec3(0,0,z0-z1), vec3(0,y1-y0,0), white_id); // right
        list[h_idx++] = new DeviceQuad(point3(x0,y1,z0), vec3(x1-x0,0,0), vec3(0,0,z1-z0), white_id); // top
        list[h_idx++] = new DeviceQuad(point3(x0,y0,z1), vec3(x1-x0,0,0), vec3(0,0,z0-z1), white_id); // bottom
    }
    // Tall box: [265,0,295] - [430,330,460]
    {
        float x0=265,y0=0,z0=295, x1=430,y1=330,z1=460;
        float theta = 3.14159265f / 4.0f;
        float c = cosf(theta);
        float s = sinf(theta);

        vec3 ax(c, 0.f, -s);
        vec3 ay(0.f, 1.f, 0.f);
        vec3 az(s, 0.f,  c);

        vec3 rx = (x1 - x0) * ax;
        vec3 ry = (y1 - y0) * ay;
        vec3 rz = (z1 - z0) * az;

        point3 center = point3((x0 + x1) * 0.5f,
                            (y0 + y1) * 0.5f,
                            (z0 + z1) * 0.5f);

        point3 origin = center - 0.5f * (rx + ry + rz);
        list[h_idx++] = new DeviceQuad(origin + rz, rx, ry, metal_id);
        list[h_idx++] = new DeviceQuad(origin + rx, -rx, ry, metal_id);
        list[h_idx++] = new DeviceQuad(origin, rz,  ry, metal_id);
        list[h_idx++] = new DeviceQuad(origin + rx + rz, -rz,  ry, metal_id);
        list[h_idx++] = new DeviceQuad(origin + ry, rx,  rz, metal_id);
        list[h_idx++] = new DeviceQuad(origin + rz, rx, -rz, metal_id);
    }

    *num_hittables_out = h_idx;
    *num_mats_out      = m_idx;
}

// ============================================================
//  Simple scene builder
// ============================================================

__global__ void kernel_build_simple_scene(DeviceHittable** list,
                                          MaterialData*    mats,
                                          int*             num_hittables_out,
                                          int*             num_mats_out) {
    int h_idx = 0;
    int m_idx = 0;

    // Materials
    int lavender_id = m_idx;
    mats[m_idx++] = {MatType::Lambertian, color(0.59f, 0.48f, 0.71f), 0.f, 0.f, color(0,0,0)};

    int ground_id = m_idx;
    mats[m_idx++] = {MatType::Lambertian, color(0.17f, 0.15f, 0.17f), 0.f, 0.f, color(0,0,0)};

    // Objects
    list[h_idx++] = new DeviceSphere(point3(0.f, 0.f, -1.f), 0.5f, lavender_id);
    list[h_idx++] = new DeviceSphere(point3(0.f, -100.5f, -1.f), 100.f, ground_id);

    *num_hittables_out = h_idx;
    *num_mats_out      = m_idx;
}

// ============================================================
//  Medium scene builder
// ============================================================

__global__ void kernel_build_medium_scene(DeviceHittable** list,
                                          MaterialData*    mats,
                                          int*             num_hittables_out,
                                          int*             num_mats_out) {
    // Reset deterministic RNG for this scene
    lcg_state = 1337u;

    int h_idx = 0;
    int m_idx = 0;

    // Materials
    int lavender_id = m_idx;
    mats[m_idx++] = {MatType::Lambertian, color(0.59f, 0.48f, 0.71f), 0.f, 0.f, color(0,0,0)};

    int ground_id = m_idx;
    mats[m_idx++] = {MatType::Lambertian, color(0.17f, 0.15f, 0.17f), 0.f, 0.f, color(0,0,0)};

    // Small random spheres
    for (int a = -5; a < 5; ++a) {
        for (int b = -5; b < 5; ++b) {
            float choose = lcg_rand();
            point3 center(a + 0.9f * lcg_rand(),
                          0.2f,
                          b + 0.9f * lcg_rand());

            // This matches your host description closely enough; kept in case
            // you later add a larger feature sphere near (4, 0.2, 0).
            vec3 diff = center - point3(4.f, 0.2f, 0.f);
            if (dot(diff, diff) <= 0.9f * 0.9f) continue;

            if (choose < 0.5f) {
                color alb(lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand());
                mats[m_idx] = {MatType::Lambertian, alb, 0.f, 0.f, color(0,0,0)};
            } else if (choose < 0.90f) {
                color alb(0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand());
                float fuzz = 0.5f * lcg_rand();
                mats[m_idx] = {MatType::Metal, alb, fuzz, 0.f, color(0,0,0)};
            } else {
                mats[m_idx] = {MatType::Dielectric, color(1.f, 1.f, 1.f), 0.f, 1.5f, color(0,0,0)};
            }

            list[h_idx++] = new DeviceSphere(center, 0.2f, m_idx++);
        }
    }

    // Main spheres
    list[h_idx++] = new DeviceSphere(point3(0.f, 0.f, -1.f), 0.5f, lavender_id);
    list[h_idx++] = new DeviceSphere(point3(0.f, -100.5f, -1.f), 100.f, ground_id);

    *num_hittables_out = h_idx;
    *num_mats_out      = m_idx;
}

// ============================================================
//  Complex scene builder
// ============================================================

__global__ void kernel_build_complex_scene(DeviceHittable** list,
                                           MaterialData*    mats,
                                           int*             num_hittables_out,
                                           int*             num_mats_out) {
    // Reset deterministic RNG for this scene
    lcg_state = 1337u;

    int h_idx = 0;
    int m_idx = 0;

    // Materials
    int lavender_id = m_idx;
    mats[m_idx++] = {MatType::Lambertian, color(0.59f, 0.48f, 0.71f), 0.f, 0.f, color(0,0,0)};

    int ground_id = m_idx;
    mats[m_idx++] = {MatType::Metal, color(0.5f, 0.5f, 0.55f), 0.f, 0.f, color(0,0,0)};

    // Dense field of small spheres
    for (int a = -10; a < 10; ++a) {
        for (int b = -10; b < 10; ++b) {
            float choose = lcg_rand();
            point3 center(a + 0.9f * lcg_rand(),
                          -0.3f,
                          b + 0.9f * lcg_rand());

            // Avoid the central lavender sphere
            if ((center - point3(0.f, 0.f, -1.f)).length() <= 0.9f) continue;

            if (choose < 0.1f) {
                color alb(lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand(),
                          lcg_rand() * lcg_rand());
                mats[m_idx] = {MatType::Lambertian, alb, 0.f, 0.f, color(0,0,0)};
            } else if (choose < 0.8f) {
                color alb(0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand(),
                          0.5f + 0.5f * lcg_rand());
                float fuzz = 0.5f * lcg_rand();
                mats[m_idx] = {MatType::Metal, alb, fuzz, 0.f, color(0,0,0)};
            } else {
                mats[m_idx] = {MatType::Dielectric, color(1.f, 1.f, 1.f), 0.f, 1.5f, color(0,0,0)};
            }

            list[h_idx++] = new DeviceSphere(center, 0.2f, m_idx++);
        }
    }

    // Main spheres
    list[h_idx++] = new DeviceSphere(point3(0.f, 0.f, -1.f), 0.5f, lavender_id);
    list[h_idx++] = new DeviceSphere(point3(0.f, -100.5f, -1.f), 100.f, ground_id);

    *num_hittables_out = h_idx;
    *num_mats_out      = m_idx;
}

// ============================================================
//  Host helpers to launch builders
// ============================================================

static DeviceScene build_scene_common(int max_objects, int max_mats,
                                       void(*builder)(DeviceHittable**, MaterialData*, int*, int*),
                                       color background) {
    DeviceScene s;
    s.background = background;

    // Allocate device arrays
    cudaMalloc(&s.d_hittables, max_objects * sizeof(DeviceHittable*));
    cudaMalloc(&s.d_materials, max_mats    * sizeof(MaterialData));

    int* d_nh; cudaMalloc(&d_nh, sizeof(int));
    int* d_nm; cudaMalloc(&d_nm, sizeof(int));

    // Build on device (single thread — no warp divergence needed)
    builder<<<1,1>>>(s.d_hittables, s.d_materials, d_nh, d_nm);
    cudaDeviceSynchronize();

    // Read back counts
    cudaMemcpy(&s.num_hittables, d_nh, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s.num_materials, d_nm, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nh);
    cudaFree(d_nm);
    return s;
}

DeviceScene build_device_random_spheres() {
    // Max: 484 small + 3 large + 1 ground = 488 objects; ~490 materials
    return build_scene_common(500, 500,
                               kernel_build_random_spheres,
                               color(0.5f, 0.7f, 1.f));
}

DeviceScene build_device_cornell_box() {
    // 6 walls + 12 box quads = 18 objects; 4 materials
    return build_scene_common(30, 10,
                               kernel_build_cornell_box,
                               color(0.f, 0.f, 0.f));
}

DeviceScene build_device_simple_scene() {
    return build_scene_common(10, 10,
                               kernel_build_simple_scene,
                               color(0.7f, 0.8f, 1.0f));
}

DeviceScene build_device_medium_scene() {
    return build_scene_common(120, 120,
                               kernel_build_medium_scene,
                               color(0.7f, 0.8f, 1.0f));
}

DeviceScene build_device_complex_scene() {
    return build_scene_common(420, 420,
                               kernel_build_complex_scene,
                               color(0.7f, 0.8f, 1.0f));
}

// Free all device allocations inside a DeviceScene.
// NOTE: Individual DeviceHittable objects allocated via device-new are freed
// by a cleanup kernel (device delete), then the pointer array itself is freed.
__global__ void kernel_free_hittables(DeviceHittable** list, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; ++i) delete list[i];
    }
}

void free_device_scene(DeviceScene& scene) {
    if (scene.d_hittables) {
        kernel_free_hittables<<<1,1>>>(scene.d_hittables, scene.num_hittables);
        cudaDeviceSynchronize();
        cudaFree(scene.d_hittables);
        scene.d_hittables = nullptr;
    }
    if (scene.d_materials) {
        cudaFree(scene.d_materials);
        scene.d_materials = nullptr;
    }
}
