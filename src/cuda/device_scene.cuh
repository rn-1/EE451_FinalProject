#pragma once

#include "../../include/vec3.h"
#include "../../include/hittable.h"

// ---- POD material descriptor for GPU (no vtable) ----
enum class MatType : int {
    Lambertian   = 0,
    Metal        = 1,
    Dielectric   = 2,
    DiffuseLight = 3
};

struct MaterialData {
    MatType type;
    color   albedo;       // Lambertian, Metal, DiffuseLight
    float   fuzz;         // Metal
    float   ir;           // Dielectric index of refraction
    color   emit_color;   // DiffuseLight
};

// hit_record for CUDA uses mat_id instead of a pointer
struct CudaHitRecord {
    point3 p;
    vec3   normal;
    float  t           = 0.f;
    bool   front_face  = false;
    int    mat_id      = -1;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// ---- Device-side Hittable base (virtual, allocated with device new) ----
class DeviceHittable {
public:
    __device__ virtual bool hit(const ray& r, float tmin, float tmax,
                                CudaHitRecord& rec) const = 0;
    __device__ virtual ~DeviceHittable() {}
};

// ---- Device-side Sphere ----
class DeviceSphere : public DeviceHittable {
public:
    point3 center;
    float  radius;
    int    mat_id;

    __device__ DeviceSphere(const point3& c, float r, int mid)
        : center(c), radius(r), mat_id(mid) {}

    __device__ bool hit(const ray& r, float tmin, float tmax,
                        CudaHitRecord& rec) const override {
        vec3  oc   = center - r.origin();
        float a    = r.direction().length_squared();
        float h    = dot(r.direction(), oc);
        float c    = oc.length_squared() - radius * radius;
        float disc = h * h - a * c;
        if (disc < 0.f) return false;

        float sqrtd = sqrtf(disc);
        float root  = (h - sqrtd) / a;
        if (root <= tmin || root >= tmax) {
            root = (h + sqrtd) / a;
            if (root <= tmin || root >= tmax) return false;
        }

        rec.t   = root;
        rec.p   = r.at(root);
        rec.mat_id = mat_id;
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};

// ---- Device-side Quad ----
class DeviceQuad : public DeviceHittable {
public:
    point3 Q;
    vec3   u, v, n, normal, w;
    float  D;
    int    mat_id;

    __device__ DeviceQuad(const point3& q, const vec3& u_, const vec3& v_, int mid)
        : Q(q), u(u_), v(v_), mat_id(mid) {
        n      = cross(u, v);
        normal = unit_vector(n);
        D      = dot(normal, Q);
        w      = n / dot(n, n);
    }

    __device__ bool hit(const ray& r, float tmin, float tmax,
                        CudaHitRecord& rec) const override {
        float denom = dot(normal, r.direction());
        if (fabsf(denom) < 1e-8f) return false;

        float t = (D - dot(normal, r.origin())) / denom;
        if (t <= tmin || t >= tmax) return false;

        point3 p    = r.at(t);
        vec3   dp   = p - Q;
        float  alpha = dot(w, cross(dp, v));
        float  beta  = dot(w, cross(u, dp));

        if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f) return false;

        rec.t      = t;
        rec.p      = p;
        rec.mat_id = mat_id;
        rec.set_face_normal(r, normal);
        return true;
    }
};

// ---- Device Scene Descriptor ----
struct DeviceScene {
    DeviceHittable** d_hittables    = nullptr;
    int              num_hittables  = 0;
    MaterialData*    d_materials    = nullptr;
    int              num_materials  = 0;
    color            background;
};

// Build random spheres scene on device (device-side new via <<<1,1>>> kernel)
DeviceScene build_device_random_spheres();

// Build Cornell Box scene on device
DeviceScene build_device_cornell_box();

DeviceScene build_device_simple_scene();
DeviceScene build_device_medium_scene();
DeviceScene build_device_complex_scene();

// Free all device allocations
void free_device_scene(DeviceScene& scene);
