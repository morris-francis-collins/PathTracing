//
//  WaveFront.metal
//  PathTracing
//
//  Created on 2/7/26.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "WaveFront.h"

using namespace metal;
using namespace raytracing;

kernel void createCameraRays(device float* accumulation,
                             
                             device float3* rayOrigins,
                             device float3* rayDirections,
                             device float3* rayThroughput,
                             device uint* pixelIndices,
                             device uint* rngDimension,
                             
                             constant uint* sobolValues,
                             
                             constant Uniforms& uniforms,
                             
                             uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    uint pixelIndex = tid.y * uniforms.width + tid.x;
    rngDimension[pixelIndex] = 0u;
    
    Sampler sampler(pixelIndex, uniforms.frameIndex, sobolValues);
    float2 pixel = static_cast<float2>(tid) + sampler.r2() - 0.5f;

    if (uniforms.frameIndex == 0u)
        setContribution(float3(0.0f), pixelIndex, accumulation);
    
    rayOrigins[pixelIndex] = uniforms.camera.position;
    rayDirections[pixelIndex] = generateRayDirection(pixel, uniforms);
    rayThroughput[pixelIndex] = float3(1.0f);
    pixelIndices[pixelIndex] = pixelIndex;
    rngDimension[pixelIndex] = sampler.dimension;
}

kernel void calculateIntersections(device float3* rayOrigins,
                                   device float3* rayDirections,
                                   device float3* rayThroughput,
                                   
                                   device IntersectionResult* intersectionResults,
                                   
                                   device uint* escapedQueue,
                                   device uint* intersectedQueue,
                                   
                                   constant uint& rayCount,
                                   device atomic_uint& escapedRayCount,
                                   device atomic_uint& intersectedRayCount,
                                   
                                   constant Textures* textures,
                                   constant Material* materials,
                                   constant int* instanceLightIndices,
                                   
                                   constant Uniforms& uniforms,
                                   device MTLAccelerationStructureInstanceDescriptor* instances,
                                   instance_acceleration_structure accelerationStructure,

                                   uint tid [[thread_position_in_grid]])
{
    if (tid >= rayCount) {
        return;
    }
    
    ray r;
    r.origin = rayOrigins[tid];
    r.direction = rayDirections[tid];
    r.min_distance = 1e-4f;
    r.max_distance = INFINITY;
        
    IntersectionResult ir = intersect<false>(r, accelerationStructure);
            
    if (ir.type == intersection_type::none) {
        uint index = atomic_fetch_add_explicit(&escapedRayCount, 1u, memory_order_relaxed);
        escapedQueue[index] = tid;
    } else {
        uint index = atomic_fetch_add_explicit(&intersectedRayCount, 1u, memory_order_relaxed);
        intersectedQueue[index] = tid;
        intersectionResults[index] = ir;
    }
}

kernel void handleEscapedRays(device atomic_float* accumulation,
                              
                              device float3* rayDirections,
                              device float3* rayThroughput,
                              device uint* pixelIndices,
                              
                              device uint* escapedQueue,
                              
                              texture2d<float> environmentMapTexture,
                                                            
                              device uint& escapedRayCount,
                              
                              constant Uniforms& uniforms,

                              uint tid [[thread_position_in_grid]])
{
    if (tid >= escapedRayCount) {
        return;
    }
    
    uint rayIndex = escapedQueue[tid];
    int environmentMapLightIndex = uniforms.environmentMapLightIndex;

    if (environmentMapLightIndex == -1)
        return;
    
    float2 uv = getEnvironmentMapUV(rayDirections[rayIndex]);
    float3 emission = environmentMapEmission(environmentMapTexture, uv);
    float3 throughput = rayThroughput[rayIndex];
    uint pixelIndex = pixelIndices[rayIndex];
        
    float3 contribution = throughput * emission;
    addContribution(contribution, pixelIndex, accumulation);
}


kernel void sampleBXDFs(device atomic_float* accumulation,
                        
                        device float3* rayOrigins,
                        device float3* rayDirections,
                        device float3* rayThroughput,
                        device uint* pixelIndices,
                        device uint* rngDimensions,
                        
                        device float3* nextRayOrigins,
                        device float3* nextRayDirections,
                        device float3* nextRayThroughput,
                        device uint* nextPixelIndices,
                        device uint* nextRngDimensions,
                        
                        device IntersectionResult* intersectionResults,
                        
                        device uint* intersectedQueue,
                        
                        device uint& intersectedCount,
                        device atomic_uint& survivedCount,
                        
                        constant uint* sobolValues,
                        constant Textures* textures,
                        constant Material* materials,
                        constant int* instanceLightIndices,
                        
                        constant Uniforms& uniforms,
                        device MTLAccelerationStructureInstanceDescriptor* instances,
                        instance_acceleration_structure accelerationStructure,
                        constant uint& bounceIndex,

                        uint tid [[thread_position_in_grid]])
{
    if (tid >= intersectedCount) {
        return;
    }

    uint rayIndex = intersectedQueue[tid];
    uint pixelIndex = pixelIndices[rayIndex];
    IntersectionResult ir = intersectionResults[tid];
    Sampler sampler(tid, uniforms.frameIndex, sobolValues); sampler.dimension = rngDimensions[rayIndex];
    
    ray r;
    r.origin = rayOrigins[rayIndex];
    r.direction = rayDirections[rayIndex];
    r.min_distance = 1e-4f;
    r.max_distance = INFINITY;
    
    auto si = getSurfaceInteraction(r, ir, instances, accelerationStructure, instanceLightIndices, textures, materials);
    
    float3 throughput = rayThroughput[rayIndex];
    float3 normal = si.normal;
    float3 position = si.position;
    float3 wi = -rayDirections[rayIndex];
    int lightIndex = si.lightIndex;
    float3 emission = si.emission;
        
    if (lightIndex != -1) {
        addContribution(throughput * emission, pixelIndex, accumulation);
    }

    SampledMaterial material = si.material;
    BSDFSample bsdfSample = sampleBXDF(wi, normal, material, Radiance, sampler.r3());

    float3 wo = bsdfSample.wo;
    float cosTheta = abs(dot(wo, normal));
    throughput *= bsdfSample.BSDF * cosTheta / bsdfSample.PDF;
    
    if (isBlack(throughput))
        return;
    
    if (bounceIndex > 4) {
        float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
        if (sampler.r() > q) {
            return;
        }
        throughput /= q;
    }
        
    uint nextRayIndex = atomic_fetch_add_explicit(&survivedCount, 1u, memory_order_relaxed);
    
    float3 offsetOrigin = position + normal * 0.001f * sign(dot(wo, normal));
    
    nextRayOrigins[nextRayIndex] = offsetOrigin;
    nextRayDirections[nextRayIndex] = wo;
    nextRayThroughput[nextRayIndex] = throughput;
    nextPixelIndices[nextRayIndex] = pixelIndex;
    nextRngDimensions[nextRayIndex] = sampler.dimension;
}
