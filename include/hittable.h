#pragma once

#include "ray.h"
#include "interval.h"

class Material;  // forward declaration

struct hit_record {
    point3    p;
    vec3      normal;
    float     t         = 0.f;
    bool      front_face = false;
    Material* mat       = nullptr;   // CPU path: raw pointer (not owned)

    HD void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    HD virtual bool hit(const ray& r,
                        interval ray_t,
                        hit_record& rec) const = 0;
    HD virtual ~Hittable() {}
};
