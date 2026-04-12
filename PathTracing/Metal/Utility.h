//
//  Utility.h
//  PathTracing
//
//  Created on 7/18/25.
//

#pragma once
#include <simd/simd.h>

#define DEBUG(...) os_log_default.log_info(__VA_ARGS__)

#define MAX_PATH_LENGTH 50
#define MAX_CAMERA_PATH_LENGTH (MAX_PATH_LENGTH + 2)
#define MAX_LIGHT_PATH_LENGTH (MAX_PATH_LENGTH + 1)

#define CAMERA_FOV_ANGLE 60.0f
#define MAX_TEXTURES 500
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

inline float3 reinhardTonemap(float3 x) {
    return x / (1.0f + x);
}

void cameraRayPDF(constant Uniforms& uniforms, float3 w, thread float& positionPDF, thread float& directionPDF);
float3 cameraWe(constant Uniforms& uniforms, float3 position);
float3 generateRayDirection(float2 pixel, constant Uniforms& uniforms);
ray generateRay(float2 pixel, constant Uniforms& uniforms);

#endif
