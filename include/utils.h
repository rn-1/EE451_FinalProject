#pragma once

#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "vec3.h"

// ---- Mathematical constants ----
static constexpr float PI       = 3.14159265358979323846f;
static constexpr float INF      = 1e30f;
static constexpr float DEG2RAD  = PI / 180.f;

HD inline float deg_to_rad(float deg) { return deg * DEG2RAD; }

// ---- Gamma-correct color to [0,255] integer ----
HD inline int to_byte(float v) {
    // gamma 2 (sqrt) correction then map to [0,255]
    v = sqrtf(v < 0.f ? 0.f : (v > 1.f ? 1.f : v));
    return static_cast<int>(255.999f * v);
}

// ---- PPM image write ----
inline void write_ppm(const std::string& filename,
                      const std::vector<color>& fb,
                      int width, int height) {
    std::ofstream out(filename);
    out << "P3\n" << width << ' ' << height << "\n255\n";
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const color& c = fb[j * width + i];
            out << to_byte(c.x()) << ' '
                << to_byte(c.y()) << ' '
                << to_byte(c.z()) << '\n';
        }
    }
}

// ---- Wall-clock timer ----
struct Timer {
    std::chrono::steady_clock::time_point t0;
    void begin() { t0 = std::chrono::steady_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---- Simple command-line argument parser ----
inline std::string get_arg(int argc, char** argv,
                            const std::string& flag,
                            const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag)
            return std::string(argv[i+1]);
    }
    return def;
}
inline bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}
