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

void cameraRayPDF(constant Camera& camera, float3 w, thread float& positionPDF, thread float& directionPDF) {
    positionPDF = 1.0f;
    float cosCamera = dot(w, camera.forward);
    directionPDF = 1.0f / (A * pow(cosCamera, 3.0f));
}

float3 cameraWe(constant Camera& camera, float3 position) {
    float3 w = normalize(position - camera.position);
    return 1.0f / (A * pow(dot(camera.forward, w), 4.0f));
}

ray generateRay(float2 pixel, const constant Uniforms& uniforms) {
    constant Camera& camera = uniforms.camera;
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;
    
    ray ray;
    ray.origin = camera.position;
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
