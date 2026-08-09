//
//  Utility.h
//  PathTracing
//
//  Created on 7/18/25.
//

#pragma once
#include <simd/simd.h>

#define DEBUG(...) os_log_default.log_info(__VA_ARGS__)

#define MAX_PATH_LENGTH 30
#define MAX_CAMERA_PATH_LENGTH (MAX_PATH_LENGTH + 2)
#define MAX_LIGHT_PATH_LENGTH (MAX_PATH_LENGTH + 1)

#define RR_MIN_BOUNCE 4
#define RR_MIN_SURVIVAL 0.05f

#define CAMERA_FOV_ANGLE 60.0f
#define MAX_TEXTURES 512
#define EPSILON 1e-3f

struct Camera {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

struct Uniforms {
    struct Camera camera;
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    unsigned int lightCount;
    int environmentMapLightIndex;
};

struct AliasEntry {
    float acceptanceProbability;
    unsigned int alias;
    float PMF;
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;
        
void debug(float x);

void debug(float3 w);

void debug(float3 w1, float3 w2);

void unimplemented();

inline float calculateEpsilon(float3 position) {
    return 1e-4f;
    return min(1e-4f * length(position), 1e-6f);
}

inline float3 calculateOffset(float3 wo, float3 n, float epsilon) {
    if (dot(wo, n) < 0.0f) n = -n;
    return wo * 0.1f * epsilon + n * epsilon;
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

inline bool isBlack(float3 w) {
    return all(w < 1e-20f);
}

inline float russianRoulette(float3 throughput, uint bounce) {
    if (bounce < RR_MIN_BOUNCE)
        return 1.0f;
    return clamp(calculateLuminance(throughput), RR_MIN_SURVIVAL, 1.0f);
}

inline float3 reinhardTonemap(float3 x) {
    return x / (1.0f + x);
}

inline void addContribution(float3 contribution, uint pixelIndex, device atomic_float* accumulation) {
    atomic_fetch_add_explicit(&accumulation[3 * pixelIndex + 0], contribution.r, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulation[3 * pixelIndex + 1], contribution.g, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulation[3 * pixelIndex + 2], contribution.b, memory_order_relaxed);
}

inline void addContribution(float3 contribution, uint pixelIndex, device float* accumulation) {
    accumulation[3 * pixelIndex + 0] += contribution.r;
    accumulation[3 * pixelIndex + 1] += contribution.g;
    accumulation[3 * pixelIndex + 2] += contribution.b;
}

inline void setContribution(float3 contribution, uint pixelIndex, device atomic_float* accumulation) {
    atomic_store_explicit(&accumulation[3 * pixelIndex + 0], contribution.r, memory_order_relaxed);
    atomic_store_explicit(&accumulation[3 * pixelIndex + 1], contribution.g, memory_order_relaxed);
    atomic_store_explicit(&accumulation[3 * pixelIndex + 2], contribution.b, memory_order_relaxed);
}

inline void setContribution(float3 contribution, uint pixelIndex, device float* accumulation) {
    accumulation[3 * pixelIndex + 0] = contribution.r;
    accumulation[3 * pixelIndex + 1] = contribution.g;
    accumulation[3 * pixelIndex + 2] = contribution.b;
}

inline float3 getFloat3FromAccumulation(uint pixelIndex, device atomic_float* accumulation) {
    return float3(atomic_load_explicit(&accumulation[3 * pixelIndex + 0], memory_order_relaxed),
                  atomic_load_explicit(&accumulation[3 * pixelIndex + 1], memory_order_relaxed),
                  atomic_load_explicit(&accumulation[3 * pixelIndex + 2], memory_order_relaxed));
}

inline float3 getFloat3FromAccumulation(uint pixelIndex, device float* accumulation) {
    return float3(accumulation[3 * pixelIndex + 0],
                  accumulation[3 * pixelIndex + 1],
                  accumulation[3 * pixelIndex + 2]);
}

void cameraRayPDF(constant Uniforms& uniforms, float3 w, thread float& positionPDF, thread float& directionPDF);
float3 cameraWe(constant Uniforms& uniforms, float3 position);
float3 generateRayDirection(float2 pixel, constant Uniforms& uniforms);
ray generateRay(float2 pixel, constant Uniforms& uniforms);

#endif
