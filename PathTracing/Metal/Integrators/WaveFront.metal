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

kernel void createCameraRays(device atomic_float* accumulation,
                             
                             device float3* rayOrigins,
                             device float3* rayDirections,
                             device float3* rayThroughput,
                             device uint* pixelIndices,
                             device uint* rngStates,
                             device bool* rayAlive,

                             device atomic_uint* rayCount,
                             
                             constant Uniforms& uniforms,
                             
                             uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    uint pixelIndex = tid.y * uniforms.width + tid.x;
    uint rng_state = init_prng(pixelIndex, uniforms.frameIndex);
    float2 pixel = static_cast<float2>(tid) + float2(prng(rng_state), prng(rng_state)) - 0.5f;
    
    rayOrigins[pixelIndex] = uniforms.camera.position;
    rayDirections[pixelIndex] = generateRayDirection(pixel, uniforms);
    rayThroughput[pixelIndex] = float3(1.0f);
    pixelIndices[pixelIndex] = pixelIndex;
    rngStates[pixelIndex] = rng_state;
    
    rayAlive[pixelIndex] = true;
}

kernel void calculateIntersections(device float3* rayOrigins,
                                   device float3* rayDirections,
                                   device float3* rayThroughput,
                                   device uint* pixelIndices,
                                   device uint* rngStates,
                                   device bool* rayAlive,
                                   
                                   device atomic_uint* rayCount,
                                   
                                   device atomic_uint* intersectionCount,
                                   device float3* intersectionPositions,
                                   device float3* intersectionNormals,
                                   device SampledMaterial* intersectionSampledMaterials,
                                   device int* intersectionLightIndices,
                                   device float3* intersectionEmission,
                                   
                                   constant Textures* textures,
                                   constant Material* materials,
                                   constant int* instanceLightIndices,
                                   
                                   device MTLAccelerationStructureInstanceDescriptor* instances,
                                   constant Uniforms& uniforms,
                                   instance_acceleration_structure accelerationStructure,
                                   
                                   uint tid [[thread_position_in_grid]])
{
    uint currentRayCount = atomic_load_explicit(rayCount, memory_order_relaxed);

    if (tid >= currentRayCount) {
        return;
    }
    
    uint rayIndex = tid;

    ray r;
    r.origin = rayOrigins[rayIndex];
    r.direction = rayDirections[rayIndex];
    r.min_distance = 0.001f;
    r.max_distance = INFINITY;
        
    IntersectionResult intres = intersect<false>(r, accelerationStructure);
            
    if (intres.type == intersection_type::none) {
        rayAlive[rayIndex] = false;
        return;
    }
    
    rayAlive[rayIndex] = true;
    
    SurfaceInteraction si = getSurfaceInteraction(r, intres, instances, accelerationStructure, instanceLightIndices, textures, materials);
    intersectionPositions[rayIndex] = si.position;
    intersectionNormals[rayIndex] = si.normal;
    intersectionSampledMaterials[rayIndex] = si.material;
    intersectionLightIndices[rayIndex] = si.lightIndex;
    intersectionEmission[rayIndex] = si.emission;
}

kernel void calculateIntersectionsWithCompaction(device float3* rayOrigins,
                                                 device float3* rayDirections,
                                                 device float3* rayThroughput,
                                                 device uint* pixelIndices,
                                                 device uint* rngStates,
                                                 device bool* rayAlive,
                                   
                                                 device float3* nextRayOrigins,
                                                 device float3* nextRayDirections,
                                                 device float3* nextRayThroughput,
                                                 device uint* nextPixelIndices,
                                                 device uint* nextRngStates,
                                                 device bool* nextRayAlive,

                                                 device atomic_uint* rayCount,
                                   
                                                 device atomic_uint* intersectionCount,
                                                 device float3* intersectionPositions,
                                                 device float3* intersectionNormals,
                                                 device SampledMaterial* intersectionSampledMaterials,
                                                 device int* intersectionLightIndices,
                                                 device float3* intersectionEmission,
                                   
                                                 constant Textures* textures,
                                                 constant Material* materials,
                                                 constant int* instanceLightIndices,
                                   
                                                 device MTLAccelerationStructureInstanceDescriptor* instances,
                                                 constant Uniforms& uniforms,
                                                 instance_acceleration_structure accelerationStructure,
                                                 
                                                 uint tid [[thread_position_in_grid]])
{
    uint currentRayCount = atomic_load_explicit(rayCount, memory_order_relaxed);

    if (tid >= currentRayCount) {
        return;
    }
    
    uint rayIndex = tid;

    ray r;
    r.origin = rayOrigins[rayIndex];
    r.direction = rayDirections[rayIndex];
    r.min_distance = 0.001f;
    r.max_distance = INFINITY;
        
    IntersectionResult intres = intersect<false>(r, accelerationStructure);
            
    if (intres.type == intersection_type::none) {
        return;
    }
    
    uint intersectionIndex = atomic_fetch_add_explicit(intersectionCount, 1, memory_order_relaxed);
    
    nextRayOrigins[intersectionIndex] = rayOrigins[rayIndex];
    nextRayDirections[intersectionIndex] = rayDirections[rayIndex];
    nextRayThroughput[intersectionIndex] = rayThroughput[rayIndex];
    nextPixelIndices[intersectionIndex] = pixelIndices[rayIndex];
    nextRngStates[intersectionIndex] = rngStates[rayIndex];
    nextRayAlive[intersectionIndex] = true;
    
    SurfaceInteraction si = getSurfaceInteraction(r, intres, instances, accelerationStructure, instanceLightIndices, textures, materials);
    // TODO: add alpha interactions and normal flipping for thin surfaces
    intersectionPositions[intersectionIndex] = si.position;
    intersectionNormals[intersectionIndex] = si.normal;
    intersectionSampledMaterials[intersectionIndex] = si.material;
    intersectionLightIndices[intersectionIndex] = si.lightIndex;
    intersectionEmission[intersectionIndex] = si.emission;
}

kernel void sampleBXDFs(device atomic_float* accumulation,
                        
                        device float3* rayOrigins,
                        device float3* rayDirections,
                        device float3* rayThroughput,
                        device uint* pixelIndices,
                        device uint* rngStates,
                        
                        device float3* nextRayOrigins,
                        device float3* nextRayDirections,
                        device float3* nextRayThroughput,
                        device uint* nextPixelIndices,
                        device uint* nextRngStates,
                        
                        device atomic_uint* rayCount,
                        device atomic_uint* nextRayCount,
                        device bool* rayAlive,
                        
                        device float3* intersectionPositions,
                        device float3* intersectionNormals,
                        device SampledMaterial* intersectionSampledMaterials,
                        device int* intersectionLightIndices,
                        device float3* intersectionEmission,
                        
                        constant Uniforms& uniforms,
                        constant uint& bounceIndex,
                        
                        uint tid [[thread_position_in_grid]])
{
    uint currentRayCount = atomic_load_explicit(rayCount, memory_order_relaxed);

    if (tid >= currentRayCount) {
        return;
    }

    uint rayIndex = tid;
    uint pixelIndex = pixelIndices[rayIndex];
    
    if (!rayAlive[rayIndex]) {
        return;
    }
    
    device uint& rng_state = rngStates[rayIndex];
    
    float3 throughput = rayThroughput[rayIndex];
    float3 normal = intersectionNormals[rayIndex];
    float3 position = intersectionPositions[rayIndex];
    float3 wi = -rayDirections[rayIndex];
    int lightIndex = intersectionLightIndices[rayIndex];
    float3 emission = intersectionEmission[rayIndex];
        
    if (lightIndex != -1) {
        addContribution(throughput * emission, pixelIndex, accumulation);
    }

    SampledMaterial material = intersectionSampledMaterials[rayIndex];
    BSDFSample bsdfSample = sampleBXDF(wi, normal, material, Radiance, float3(prng(rng_state), prng(rng_state), prng(rng_state)));
    
    if (bsdfSample.PDF <= 0.0f || isBlack(bsdfSample.BSDF)) {
        rayAlive[rayIndex] = false;
        return;
    }

    float3 wo = bsdfSample.wo;
    float cosTheta = abs(dot(wo, normal));
    float3 newThroughput = throughput * bsdfSample.BSDF * cosTheta / bsdfSample.PDF;
    
    if (bounceIndex > 4) {
        float q = clamp(calculateLuminance(newThroughput), 0.01f, 1.0f);
        if (prng(rng_state) > q) {
            rayAlive[rayIndex] = false;
            return;
        }
        newThroughput /= q;
    }
        
    uint nextRayIndex = atomic_fetch_add_explicit(nextRayCount, 1, memory_order_relaxed);
    
    float3 offsetOrigin = position + normal * 0.001f * sign(dot(wo, normal));
    
    nextRayOrigins[nextRayIndex] = offsetOrigin;
    nextRayDirections[nextRayIndex] = wo;
    nextRayThroughput[nextRayIndex] = newThroughput;
    nextPixelIndices[nextRayIndex] = pixelIndex;
    nextRngStates[nextRayIndex] = rng_state;
}
