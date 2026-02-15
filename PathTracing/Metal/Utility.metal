//
//  Utility.metal
//  PathTracing
//
//  Created on 7/18/25.
//

#include <metal_stdlib>
#include "Utility.h"

using namespace metal;
using namespace raytracing;

// MARK: Cameras

void cameraRayPDF(constant Uniforms& uniforms, float3 w, thread float& positionPDF, thread float& directionPDF) {
    positionPDF = 1.0f;
    float cosCamera = dot(w, uniforms.camera.forward);
    float A = 4.0f * (uniforms.width / uniforms.height) * pow(tan(M_PI_F * CAMERA_FOV_ANGLE * 0.5f / 180.0f), 2.0f);
    directionPDF = 1.0f / (A * pow(cosCamera, 3.0f));
}

float3 cameraWe(constant Uniforms& uniforms, float3 position) {
    float3 w = normalize(position - uniforms.camera.position);
    float A = 4.0f * (uniforms.width / uniforms.height) * pow(tan(M_PI_F * CAMERA_FOV_ANGLE * 0.5f / 180.0f), 2.0f);
    return 1.0f / (A * pow(dot(uniforms.camera.forward, w), 4.0f));
}

float3 generateRayDirection(float2 pixel, constant Uniforms& uniforms) {
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;
    return normalize(uv.x * uniforms.camera.right + uv.y * uniforms.camera.up + uniforms.camera.forward);
}

ray generateRay(float2 pixel, constant Uniforms& uniforms) {
    ray ray;
    ray.origin = uniforms.camera.position;
    ray.direction = generateRayDirection(pixel, uniforms);
    ray.min_distance = 1e-6f;
    ray.max_distance = INFINITY;
    return ray;
}

void debug(float x) {
    os_log_default.log_info("%f", x);
}

void debug(float3 w) {
    os_log_default.log_info("mag : %f : float3(%f, %f, %f)", length(w), w.x, w.y, w.z);
}

void debug(float3 w1, float3 w2) {
    os_log_default.log_info("v1: mag %f float3(%f, %f, %f), v2: mag %f float3(%f, %f, %f)", length(w1), w1.x, w1.y, w1.z, length(w2), w2.x, w2.y, w2.z);
}

void unimplemented() {
    os_log_default.log_info("unimplemented.");
}
