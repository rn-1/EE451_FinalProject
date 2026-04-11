#pragma once

#include "hittable.h"
#include <vector>
#include <memory>

// CPU-side scene container using shared_ptr
class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    HittableList() {}
    void add(std::shared_ptr<Hittable> obj) { objects.push_back(obj); }
    void clear() { objects.clear(); }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record tmp;
        bool hit_anything = false;
        float closest = ray_t.tmax;

        for (const auto& obj : objects) {
            if (obj->hit(r, interval(ray_t.tmin, closest), tmp)) {
                hit_anything = true;
                closest = tmp.t;
                rec = tmp;
            }
        }
        return hit_anything;
    }
};
