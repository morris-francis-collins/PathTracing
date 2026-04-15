//
//  SPPM.metal
//  PathTracing
//
//  Created on 4/12/26.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "SPPM.h"

using namespace metal;
using namespace raytracing;

kernel void createCameraRaysSPPM(device float3* rayOrigins,
                                 device float3* rayDirections,
                                 device uint* rngStates,
                                 
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
    rngStates[pixelIndex] = rng_state;
}

kernel void generateHitPointsSPPM(device float* accumulation,
                                  
                                  device float3* rayOrigins,
                                  device float3* rayDirections,
                                  device uint* rngStates,
                                  
                                  constant Light* lights,
                                  constant LightTriangle* lightTriangles,
                                  constant int* instanceLightIndices,
                                  constant AliasEntry* lightAliasEntries,
                                  constant AliasEntry* lightTriangleAliasEntries,
                                  texture2d<float> environmentMapTexture,
                                  constant AliasEntry* environmentMapAliasEntries,
                                                
                                  constant Material* materials,
                                  constant Textures* textures,
                                  
                                  device float3* hitPointBSDFs,
                                  device float3* hitPointLocations,
                                  device float3* hitPointIncomingDirections,
                                  device float3* hitPointNormals,
                                  device uint* hitPointHashes,
                                  
                                  device atomic_uint* hashTableCounts,
                                  constant float& hashGridSize,
                                  
                                  constant Uniforms& uniforms,
                                  device MTLAccelerationStructureInstanceDescriptor* instances,
                                  instance_acceleration_structure accelerationStructure,
                                  
                                  uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    uint pixelIndex = tid.y * uniforms.width + tid.x;
    device uint& rng_state = rngStates[pixelIndex];
    Sampler sampler(pixelIndex, uniforms.frameIndex);

    ray ray;
    ray.origin = rayOrigins[pixelIndex];
    ray.direction = rayDirections[pixelIndex];
    ray.min_distance = 1e-4f;
    ray.max_distance = INFINITY;
    
    float3 throughput = float3(1.0f);
    
    if (uniforms.frameIndex == 0u) {
        accumulation[3 * pixelIndex + 0] = 0;
        accumulation[3 * pixelIndex + 1] = 0;
        accumulation[3 * pixelIndex + 2] = 0;
    }
            
    for (uint bounce = 0; bounce < MAX_PATH_LENGTH; bounce++) {
        IntersectionResult ir = intersect<false>(ray, accelerationStructure);
        
        if (ir.type == intersection_type::none) {
            float2 uv = getEnvironmentMapUV(ray.direction);
            float3 emission = environmentMapEmission(environmentMapTexture, uv);
            float3 contribution = throughput * emission;
            
            accumulation[3 * pixelIndex + 0] += contribution.r;
            accumulation[3 * pixelIndex + 1] += contribution.g;
            accumulation[3 * pixelIndex + 2] += contribution.b;

            break;
        }
                
        SurfaceInteraction si = getSurfaceInteraction(ray, ir, instances, accelerationStructure, instanceLightIndices, textures, materials);
        SampledMaterial material = si.material;
        
        if (si.hitLight()) {
            float3 contribution = throughput * si.emission;

            accumulation[3 * pixelIndex + 0] += contribution.r;
            accumulation[3 * pixelIndex + 1] += contribution.g;
            accumulation[3 * pixelIndex + 2] += contribution.b;
            
            break;
        }
        
        if (!material.isPerfectSpecular()) {
            uint hashedLocation = hashLocation(si.position, hashGridSize);

            hitPointBSDFs[pixelIndex] = throughput;
            hitPointLocations[pixelIndex] = si.position;
            hitPointIncomingDirections[pixelIndex] = -ray.direction;
            hitPointNormals[pixelIndex] = si.normal;
            hitPointHashes[pixelIndex] = hashedLocation;

            atomic_fetch_add_explicit(&hashTableCounts[hashedLocation], 1u, memory_order_relaxed);
            return;
        }
        
        BSDFSample bs = sampleBXDF(-ray.direction, si.normal, material, Radiance, float3(prng(rng_state), prng(rng_state), prng(rng_state)));
        
        if (!bs.delta) {
            float selectionPDF;
            constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
            LightSample lightSample = sampleLight(si.position, light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);
            
            float3 wi = -ray.direction, wo = lightSample.wo;
            float3 pos1 = si.position, pos2 = lightSample.position;
            float distance = lightSample.distance;
            
            float cosCamera = dot(wo, si.normal);
            float cosLight = light.type == AREA_LIGHT ? dot(-wo, si.normal) : 1.0f;
        
            if (cosCamera > 0.0f and cosLight > 0.0f and isVisible(pos1, si.normal, pos2, lightSample.normal, instances, accelerationStructure)) {
                float3 BSDF = getBXDF(wi, wo, si.normal, material, Radiance);

                float G = cosCamera * cosLight / (distance * distance);
                float lightPDF = selectionPDF * lightSample.PDF;
                
                float3 contribution;

                if (light.isDelta()) {
                    contribution = throughput * BSDF * lightSample.emission * G / lightPDF;
                } else {
                    float bsdfPDF = getPDF(wi, wo, si.normal, material);
                    float weight = powerHeuristic(lightPDF, bsdfPDF);
                    contribution = throughput * BSDF * lightSample.emission * G * weight / lightPDF;
                }
                
                accumulation[3 * pixelIndex + 0] += contribution.r;
                accumulation[3 * pixelIndex + 1] += contribution.g;
                accumulation[3 * pixelIndex + 2] += contribution.b;
            }
        }
                        
        throughput *= bs.BSDF * abs(dot(bs.wo, si.normal)) / bs.PDF;
        
        if (isBlack(throughput))
            break;
                
        ray.origin = si.position + bs.wo * 1e-4f;
        ray.direction = bs.wo;
    }
    
    hitPointBSDFs[pixelIndex] = float3(0.0f); // when we cant find a diffuse surface or escape
    hitPointLocations[pixelIndex] = float3(0.0f);
    hitPointHashes[pixelIndex] = 0u;
}

kernel void createHashGridSPPM(device float3* hitPointBSDFs,
                               device float3* hitPointLocations,
                               device float3* hitPointIncomingDirections,
                               device float3* hitPointNormals,
                               device uint* hitPointHashes,

                               device float3* hashTableBSDFs,
                               device float3* hashTableLocations,
                               device float3* hashTableIncomingDirections,
                               device float3* hashTableNormals,
                               device uint2* hashTableShadingPixels,
                               
                               device uint* totalPhotonCounts,
                               device uint* currentPhotonCounts,
                               device float* gatheringRadii,

                               device uint* hashTableOffsets,
                               device atomic_uint* hashTableIndices,
                               device atomic_float& newHashGridSize,
                               
                               constant Uniforms& uniforms,
                               
                               uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    uint pixelIndex = tid.y * uniforms.width + tid.x;
        
    if (all(hitPointBSDFs[pixelIndex] == 0.0f))
        return;
    
    uint hash = hitPointHashes[pixelIndex];
    uint hashTableIndex = hashTableOffsets[hash] + atomic_fetch_add_explicit(&hashTableIndices[hash], 1u, memory_order_relaxed);
    
    hashTableBSDFs[hashTableIndex] = hitPointBSDFs[pixelIndex];
    hashTableLocations[hashTableIndex] = hitPointLocations[pixelIndex];
    hashTableIncomingDirections[hashTableIndex] = hitPointIncomingDirections[pixelIndex];
    hashTableNormals[hashTableIndex] = hitPointNormals[pixelIndex];
    hashTableShadingPixels[hashTableIndex] = tid;
    
    uint N = max(1u, totalPhotonCounts[pixelIndex]);
    uint M = currentPhotonCounts[pixelIndex];
    
    gatheringRadii[pixelIndex] *= sqrt((N + ALPHA * M) / (N + M));
    
    uint radiusBits = as_type<uint>(gatheringRadii[pixelIndex]); // reinterpret since no atomic max float
    atomic_fetch_max_explicit((device atomic_uint*)&newHashGridSize, radiusBits, memory_order_relaxed);
    
    totalPhotonCounts[pixelIndex] += M;
    currentPhotonCounts[pixelIndex] = 0u;
}

kernel void tracePhotonsSPPM(device atomic_float* accumulation,
                             
                             constant Light* lights,
                             constant LightTriangle* lightTriangles,
                             constant int* instanceLightIndices,
                             constant AliasEntry* lightAliasEntries,
                             constant AliasEntry* lightTriangleAliasEntries,
                             texture2d<float> environmentMapTexture,
                             constant AliasEntry* environmentMapAliasEntries,
                             
                             constant Material* materials,
                             constant Textures* textures,
                             
                             device float3* hashTableBSDFs,
                             device float3* hashTableLocations,
                             device float3* hashTableIncomingDirections,
                             device float3* hashTableNormals,
                             device uint2* hashTableShadingPixels,
                              
                             device uint* hashTableOffsets,
                             device uint* hashTableCounts,
                             constant float& hashGridSize,
                             
                             device atomic_uint* currentPhotonCounts,
                             device float* gatheringRadii,

                             constant Uniforms& uniforms,
                             device MTLAccelerationStructureInstanceDescriptor* instances,
                             instance_acceleration_structure accelerationStructure,

                             uint tid [[thread_position_in_grid]])
{
    if (tid >= PHOTON_COUNT)
        return;

    Sampler sampler(tid, uniforms.frameIndex);

    float selectionPDF;
    constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
    LightEmissionSample les = sampleLightEmission(light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);
    
    float3 throughput = les.emission / (selectionPDF * les.positionPDF * les.directionPDF);
    
    ray ray;
    ray.origin = les.position + calculateOffset(les.wo, les.normal, 1e-4f);
    ray.direction = les.wo;
    ray.min_distance = 1e-4f;
    ray.max_distance = INFINITY;
    
    if (light.type == AREA_LIGHT)
        throughput *= abs(dot(les.normal, ray.direction));
    
    for (uint bounce = 0; bounce < MAX_PATH_LENGTH; bounce++) {
        IntersectionResult ir = intersect<false>(ray, accelerationStructure);
        
        if (ir.type == intersection_type::none) {
            break;
        }
                
        SurfaceInteraction si = getSurfaceInteraction(ray, ir, instances, accelerationStructure, instanceLightIndices, textures, materials);
        SampledMaterial material = si.material;

        if (!material.isPerfectSpecular()) {
            float3 photonPosition = si.position;
            int3 cell = int3(floor(photonPosition / hashGridSize));
            
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int3 neighbor = cell + int3(dx, dy, dz);
                        uint h = clamp(hashCell(neighbor), 0u, HASH_TABLE_SIZE - 1);
                        
                        for (uint i = hashTableOffsets[h]; i < hashTableOffsets[h] + hashTableCounts[h]; i++) {
                            uint2 shadingPixel = hashTableShadingPixels[i];
                            uint shadingIndex = shadingPixel.y * uniforms.width + shadingPixel.x;
                            
                            float3 cameraPosition = hashTableLocations[i];
                            float r = gatheringRadii[shadingIndex];
                            float r2 = r * r;
                            float d2 = length_squared(photonPosition - cameraPosition);
                            
                            if (d2 < r2) {
                                float3 wi = hashTableIncomingDirections[i];
                                float3 wo = -ray.direction;
                                float3 n = hashTableNormals[i];
                                
                                float3 contribution = hashTableBSDFs[i] * throughput * getBXDF(wi, wo, n, material, Radiance) * epanechnikov(d2, r2) / float(PHOTON_COUNT);

                                atomic_fetch_add_explicit(&accumulation[3 * shadingIndex + 0], contribution.r, memory_order_relaxed);
                                atomic_fetch_add_explicit(&accumulation[3 * shadingIndex + 1], contribution.g, memory_order_relaxed);
                                atomic_fetch_add_explicit(&accumulation[3 * shadingIndex + 2], contribution.b, memory_order_relaxed);
                                
                                atomic_fetch_add_explicit(&currentPhotonCounts[shadingIndex], 1u, memory_order_relaxed);
                            }
                        }
                    }
                }
            }
        }
        
        if (si.hitLight()) {
            break;
        }

        BSDFSample bs = sampleBXDF(-ray.direction, si.normal, material, Radiance, sampler.r3());

        throughput *= bs.BSDF * abs(dot(bs.wo, si.normal)) / bs.PDF;
        
        if (isBlack(throughput))
            break;
        if (bounce > 4) {
            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
            if (sampler.r() > q) break;
            throughput /= q;
        }
                
        ray.origin = si.position + bs.wo * 1e-4f;
        ray.direction = bs.wo;
    }
}

kernel void finalizeAccumulationSPPM(uint2 tid [[thread_position_in_grid]],
                                     device float* accumulation,
                                     constant Uniforms& uniforms,
                                     texture2d<float, access::read_write> finalImage)
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    
    
    float3 contribution = float3(accumulation[3 * pixelIndex + 0],
                                 accumulation[3 * pixelIndex + 1],
                                 accumulation[3 * pixelIndex + 2]);
    
    float3 color = contribution / (uniforms.frameIndex + 1);
    finalImage.write(float4(color, 1.0f), tid);
}
