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

#define PIXEL_WIDTH 800.0f
#define PIXEL_HEIGHT 600.0f
#define ASPECT_RATIO (PIXEL_WIDTH / PIXEL_HEIGHT)
#define A 4.0f * ASPECT_RATIO * pow(tan(M_PI_F * CAMERA_FOV_ANGLE * 0.5f / 180.0f), 2.0f)

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

inline float isBlack(float3 w) {
    return all(w < 1e-20f);
}

void cameraRayPDF(const constant Camera& camera, float3 w, thread float& positionPDF, thread float& directionPDF);
float3 cameraWe(constant Camera& camera, float3 position);
ray generateRay(float2 pixel, const constant Uniforms& uniforms);

#endif
