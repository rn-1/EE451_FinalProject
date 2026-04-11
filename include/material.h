#pragma once

#include "ray.h"
#include "hittable.h"

// ---- Abstract Material (CPU path) ----
class Material {
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec,
                         color& attenuation, ray& scattered,
                         std::mt19937& rng) const = 0;
    virtual color emitted() const { return color(0.f, 0.f, 0.f); }
    virtual ~Material() {}
};

// ---- Lambertian (diffuse) ----
class Lambertian : public Material {
public:
    color albedo;
    explicit Lambertian(const color& a) : albedo(a) {}

    bool scatter(const ray& /*r_in*/, const hit_record& rec,
                 color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        vec3 scatter_dir = rec.normal + random_unit_vector(rng);
        if (scatter_dir.near_zero()) scatter_dir = rec.normal;
        scattered   = ray(rec.p, scatter_dir);
        attenuation = albedo;
        return true;
    }
};

// ---- Metal (specular with fuzz) ----
class Metal : public Material {
public:
    color albedo;
    float fuzz;
    Metal(const color& a, float f) : albedo(a), fuzz(f < 1.f ? f : 1.f) {}

    bool scatter(const ray& r_in, const hit_record& rec,
                 color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        reflected = unit_vector(reflected) + fuzz * random_unit_vector(rng);
        scattered   = ray(rec.p, reflected);
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0.f;
    }
};

// ---- Dielectric (glass) ----
class Dielectric : public Material {
public:
    float ir;  // index of refraction
    explicit Dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    bool scatter(const ray& r_in, const hit_record& rec,
                 color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        attenuation = color(1.f, 1.f, 1.f);
        float ratio = rec.front_face ? (1.f / ir) : ir;
        vec3  unit_dir = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_dir, rec.normal), 1.f);
        float sin_theta = sqrtf(1.f - cos_theta * cos_theta);

        bool cannot_refract = ratio * sin_theta > 1.f;
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, ratio) > rand_float(rng))
            direction = reflect(unit_dir, rec.normal);
        else
            direction = refract(unit_dir, rec.normal, ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    static float reflectance(float cosine, float ref_idx) {
        // Schlick approximation
        float r0 = (1.f - ref_idx) / (1.f + ref_idx);
        r0 *= r0;
        return r0 + (1.f - r0) * powf(1.f - cosine, 5.f);
    }
};

// ---- Diffuse Light (emissive) ----
class DiffuseLight : public Material {
public:
    color emit;
    explicit DiffuseLight(const color& c) : emit(c) {}

    bool scatter(const ray&, const hit_record&, color&, ray&,
                 std::mt19937&) const override {
        return false;  // lights do not scatter
    }
    color emitted() const override { return emit; }
};
