#pragma once

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#include <curand_kernel.h>
#define HD __host__ __device__
#else
#define HD
#include <random>
#endif

class vec3 {
public:
    float e[3];

    HD vec3() : e{0.f, 0.f, 0.f} {}
    HD vec3(float x, float y, float z) : e{x, y, z} {}

    HD float x() const { return e[0]; }
    HD float y() const { return e[1]; }
    HD float z() const { return e[2]; }

    HD vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    HD float operator[](int i) const { return e[i]; }
    HD float& operator[](int i) { return e[i]; }

    HD vec3& operator+=(const vec3& v) {
        e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2];
        return *this;
    }
    HD vec3& operator*=(float t) {
        e[0] *= t; e[1] *= t; e[2] *= t;
        return *this;
    }
    HD vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0]; e[1] *= v.e[1]; e[2] *= v.e[2];
        return *this;
    }
    HD vec3& operator/=(float t) { return *this *= (1.f / t); }

    HD float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    HD float length() const { return sqrtf(length_squared()); }

    HD bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
    }
};

using color  = vec3;
using point3 = vec3;

// --- Arithmetic operators ---

HD inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}
HD inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]);
}
HD inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}
HD inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
HD inline vec3 operator*(const vec3& v, float t) { return t * v; }
HD inline vec3 operator/(const vec3& v, float t) { return (1.f/t) * v; }

HD inline float dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}
HD inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1]*v.e[2] - u.e[2]*v.e[1],
                u.e[2]*v.e[0] - u.e[0]*v.e[2],
                u.e[0]*v.e[1] - u.e[1]*v.e[0]);
}
HD inline vec3 unit_vector(const vec3& v) { return v / v.length(); }

HD inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.f * dot(v, n) * n;
}

HD inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.f);
    vec3 r_out_perp  = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// --- Random helpers ---

#ifdef __CUDACC__
// CUDA path: use cuRAND
__device__ inline float rand_float(curandState* rs) {
    return curand_uniform(rs);
}
__device__ inline float rand_float(curandState* rs, float lo, float hi) {
    return lo + (hi - lo) * curand_uniform(rs);
}
__device__ inline vec3 random_vec(curandState* rs) {
    return vec3(curand_uniform(rs), curand_uniform(rs), curand_uniform(rs));
}
__device__ inline vec3 random_vec(curandState* rs, float lo, float hi) {
    return vec3(rand_float(rs, lo, hi),
                rand_float(rs, lo, hi),
                rand_float(rs, lo, hi));
}
__device__ inline vec3 random_in_unit_sphere(curandState* rs) {
    while (true) {
        vec3 p = random_vec(rs, -1.f, 1.f);
        if (p.length_squared() < 1.f) return p;
    }
}
__device__ inline vec3 random_unit_vector(curandState* rs) {
    return unit_vector(random_in_unit_sphere(rs));
}
__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* rs) {
    vec3 v = random_unit_vector(rs);
    return (dot(v, normal) > 0.f) ? v : -v;
}
__device__ inline vec3 random_in_unit_disk(curandState* rs) {
    while (true) {
        vec3 p(rand_float(rs, -1.f, 1.f), rand_float(rs, -1.f, 1.f), 0.f);
        if (p.length_squared() < 1.f) return p;
    }
}

#else
// CPU path: use <random>
inline float rand_float(std::mt19937& rng) {
    static std::uniform_real_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}
inline float rand_float(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}
inline vec3 random_vec(std::mt19937& rng) {
    return vec3(rand_float(rng), rand_float(rng), rand_float(rng));
}
inline vec3 random_vec(std::mt19937& rng, float lo, float hi) {
    return vec3(rand_float(rng, lo, hi),
                rand_float(rng, lo, hi),
                rand_float(rng, lo, hi));
}
inline vec3 random_in_unit_sphere(std::mt19937& rng) {
    while (true) {
        vec3 p = random_vec(rng, -1.f, 1.f);
        if (p.length_squared() < 1.f) return p;
    }
}
inline vec3 random_unit_vector(std::mt19937& rng) {
    return unit_vector(random_in_unit_sphere(rng));
}
inline vec3 random_on_hemisphere(const vec3& normal, std::mt19937& rng) {
    vec3 v = random_unit_vector(rng);
    return (dot(v, normal) > 0.f) ? v : -v;
}
inline vec3 random_in_unit_disk(std::mt19937& rng) {
    while (true) {
        vec3 p(rand_float(rng, -1.f, 1.f), rand_float(rng, -1.f, 1.f), 0.f);
        if (p.length_squared() < 1.f) return p;
    }
}
#endif

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
