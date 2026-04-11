#pragma once

#include "hittable.h"
#include "material.h"

class Sphere : public Hittable {
public:
    point3    center;
    float     radius;
    Material* mat;   // raw non-owning pointer (for CUDA device-new compat)

    HD Sphere(const point3& c, float r, Material* m)
        : center(c), radius(r), mat(m) {}

    HD bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3  oc  = center - r.origin();
        float a   = r.direction().length_squared();
        float h   = dot(r.direction(), oc);
        float c   = oc.length_squared() - radius * radius;
        float disc = h * h - a * c;

        if (disc < 0.f) return false;

        float sqrtd = sqrtf(disc);
        float root  = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) return false;
        }

        rec.t  = root;
        rec.p  = r.at(root);
        rec.mat = mat;
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};
