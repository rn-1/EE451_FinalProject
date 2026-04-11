#pragma once

#include <cfloat>

struct interval {
    float tmin, tmax;

    HD interval() : tmin(FLT_MAX), tmax(-FLT_MAX) {}
    HD interval(float lo, float hi) : tmin(lo), tmax(hi) {}

    HD bool contains(float t)     const { return tmin <= t && t <= tmax; }
    HD bool surrounds(float t)    const { return tmin <  t && t <  tmax; }
    HD float clamp(float t)       const {
        if (t < tmin) return tmin;
        if (t > tmax) return tmax;
        return t;
    }

    static const interval empty;
    static const interval universe;
};

inline const interval interval::empty    = interval( FLT_MAX, -FLT_MAX);
inline const interval interval::universe = interval(-FLT_MAX,  FLT_MAX);
