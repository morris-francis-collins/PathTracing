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

void cameraRayPDF(const constant Camera& camera, float3 w, thread float& positionPDF, thread float& directionPDF) {
    positionPDF = 1.0f;
    directionPDF = 1.0f / (A * pow(dot(w, camera.forward), 3.0f));
}

ray generateRay(float2 pixel, const constant Uniforms& uniforms) {
    constant Camera& camera = uniforms.camera;
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;

    ray ray;
    ray.origin = camera.position + camera.forward * 1e-4f;
    ray.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
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
