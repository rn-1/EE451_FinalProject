#pragma once

#include "hittable.h"
#include "material.h"

// A planar quadrilateral defined by corner Q and edge vectors u, v.
// The quad covers points Q + s*u + t*v  for s,t in [0,1].
class Quad : public Hittable {
public:
    point3    Q;      // corner
    vec3      u, v;   // edge vectors
    Material* mat;

    // Precomputed:
    vec3  n;          // cross(u, v)
    vec3  normal;     // unit normal
    float D;          // plane constant: dot(normal, Q)
    vec3  w;          // n / dot(n, n)  — for UV projection

    HD Quad(const point3& q, const vec3& u_, const vec3& v_, Material* m)
        : Q(q), u(u_), v(v_), mat(m) {
        n      = cross(u, v);
        normal = unit_vector(n);
        D      = dot(normal, Q);
        w      = n / dot(n, n);
    }

    HD bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        float denom = dot(normal, r.direction());
        // Ray parallel to plane
        if (fabsf(denom) < 1e-8f) return false;

        float t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.surrounds(t)) return false;

        // Check if hit point is within the quad bounds using planar coords
        point3 p    = r.at(t);
        vec3   dp   = p - Q;
        float  alpha = dot(w, cross(dp, v));
        float  beta  = dot(w, cross(u, dp));

        if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f)
            return false;

        rec.t   = t;
        rec.p   = p;
        rec.mat = mat;
        rec.set_face_normal(r, normal);
        return true;
    }
};
