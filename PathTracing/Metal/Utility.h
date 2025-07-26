//
//  Utility.h
//  PathTracing
//
//  Created on 7/18/25.
//

#pragma once
#include <simd/simd.h>

#define DEBUG(...) os_log_default.log_info(__VA_ARGS__)

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_SPHERE   2
#define GEOMETRY_MASK_LIGHT    4
#define GEOMETRY_MASK_TRANSPARENT 8
#define GEOMETRY_MASK_OPAQUE 16

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_SPHERE)

#define RAY_MASK_PRIMARY   (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_TRANSPARENT | GEOMETRY_MASK_OPAQUE)
#define RAY_MASK_SHADOW    GEOMETRY_MASK_OPAQUE | GEOMETRY_MASK_LIGHT
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_TRANSPARENT | GEOMETRY_MASK_OPAQUE

#define CAMERA_FOV_ANGLE 60.0f
#define MAX_TEXTURES 120
#define EPSILON 1e-3f

struct Camera {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

struct Uniforms {
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    struct Camera camera;
    unsigned int lightCount;
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

constant unsigned int primes[] =
{
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
    811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
    1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373,
    1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657,
    1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
    1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129,
    2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267
};

struct HaltonSampler {
    uint index;
    uint dimension;
    uint frameIndex;
    
    HaltonSampler(uint _index, uint _frameIndex) {
        index = _index;
        dimension = 0;
        frameIndex = _frameIndex;
    }
        
    float halton(uint i, uint d) {
        uint b = primes[d];

        float f = 1.0f;
        float invB = 1.0f / b;

        float r = 0;

        while (i > 0)
        {
            f = f * invB;
            r = r + f * (i % b);
            i = i / b;
        }
        
        return r;
    }

    uint hash(uint input) {
        uint state = input * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    float scrambledHalton(uint index, uint dimension, uint frameIndex) {
        return fract(halton(index, dimension) + float(hash(dimension + frameIndex * 719)) / 4294967296.0f);
    }
    
    float r() {
        return scrambledHalton(index, dimension++, frameIndex);
    }
    
    float2 r2() {
        return float2(r(), r());
    }
    
    float3 r3() {
        return float3(r(), r(), r());
    }
};
        
void debug(float x);

void debug(float3 w);

inline float calculateEpsilon(float3 position) {
    return min(1e-4f * length(position), 1e-6f);
}

inline float3 calculateOffset(float3 wo, float3 n, float epsilon) {
    if (dot(wo, n) < 0.0f) n = -n;
    return wo * 0.1f * epsilon + n * epsilon;
}

inline float halton(uint i, uint d) {
        uint b = primes[d];

        float f = 1.0f;
        float invB = 1.0f / b;

        float r = 0;

        while (i > 0)
        {
            f = f * invB;
            r = r + f * (i % b);
            i = i / b;
        }
        
        return r;
}

inline uint hash(uint input) {
        uint state = input * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

inline float scrambledHalton(uint index, uint dimension, uint frameIndex) {
    return fract(halton(index, dimension) + float(hash(dimension + frameIndex * 719)) / 4294967296.0f);
}

inline float calculateLuminance(float3 w) {
    return dot(w, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float balanceHeuristic(float main, float other) {
    return main / (main + other);
}

inline float powerHeuristic(float main, float other) {
    float main2 = main * main;
    float other2 = other * other;
    return main2 / (main2 + other2);
}

#endif
