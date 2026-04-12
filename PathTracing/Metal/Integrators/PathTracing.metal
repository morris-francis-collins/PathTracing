//
//  PathTracing.metal
//  PathTracing
//
//  Created on 4/11/26.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "PathTracing.h"

using namespace metal;
using namespace raytracing;

float3 pathIntegrator(float2 pixel,
                      constant Uniforms& uniforms,
                      device MTLAccelerationStructureInstanceDescriptor *instances,
                      instance_acceleration_structure accelerationStructure,
                      constant Light *lights,
                      constant LightTriangle *lightTriangles,
                      constant int* instanceLightIndices,
                      thread Sampler& sampler,
                      constant Textures* textures,
                      constant Material* materials,
                      constant AliasEntry* lightAliasEntries,
                      constant AliasEntry* lightTriangleAliasEntries,
                      texture2d<float> environmentMapTexture,
                      constant AliasEntry* environmentMapAliasEntries)
{
    ray ray = generateRay(pixel, uniforms);

    float3 contribution = float3(0.0f);
    float3 throughput = float3(1.0f);
    float PDF = 0.0f;
    bool prevSpecular = true;
    
    bool inMedium = false;
    float attenuationDistance = 0.0f;
    
    for (int bounce = 0; bounce < MAX_PATH_LENGTH; bounce++) {
        IntersectionResult intersection = intersect<false>(ray, accelerationStructure);
        
        if (intersection.type == intersection_type::none) {
            int environmentMapLightIndex = uniforms.environmentMapLightIndex;

            if (environmentMapLightIndex == -1)
                break;
            
            float2 uv = getEnvironmentMapUV(ray.direction);
            float3 emission = environmentMapEmission(environmentMapTexture, uv);

            if (prevSpecular) {
                contribution += throughput * emission;
            } else {
                float lightPDF = environmentLightSamplePDF(lights[environmentMapLightIndex], environmentMapAliasEntries, uv);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += throughput * emission * weight;
            }

            break;
        }
        
        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, instances, accelerationStructure, instanceLightIndices, textures, materials);
        SampledMaterial material = surfaceInteraction.material;
        
        if (material.BXDFs == DIFFUSE || material.BXDFs == CONDUCTOR || material.BXDFs == DIELECTRIC_REFLECTION) { // TODO: move to surface interaction
            if (dot(-ray.direction, surfaceInteraction.normal) < 0.0f)
                surfaceInteraction.normal = -surfaceInteraction.normal;
        }
        
        if (material.alphaMode == ALPHA_MASK && material.alpha < material.alphaCutoff) {
            ray.origin = surfaceInteraction.position + ray.direction * 1e-4f;
            continue;
        }
        
        if (material.alphaMode == ALPHA_BLEND && sampler.r() > material.alpha) {
            ray.origin = surfaceInteraction.position + ray.direction * 1e-4f;
            continue;
        }

        float3 n = surfaceInteraction.normal;

        if (surfaceInteraction.hitLight()) {
            constant Light& light = lights[surfaceInteraction.lightIndex];
            float3 color = surfaceInteraction.emission;

            if (prevSpecular) {
                contribution += throughput * color;
            } else {
                float lightPDF = getLightSelectionPDF(light, lightAliasEntries) * getLightSamplePDF(light);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += throughput * color * weight;
            }
        }
        
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, sampler.r3());
        
        PDF = bsdfSample.PDF;
        float3 wo = bsdfSample.wo;
        inMedium ^= bsdfSample.transmitted;
        float epsilon = calculateEpsilon(surfaceInteraction.position);
        prevSpecular = bsdfSample.delta;
        
//        if (inMedium) {
//            attenuationDistance += length(surfaceInteraction.position - ray.origin);
//        } else if (bsdfSample.transmitted) {
//            float3 absorption = log(1 - material.color);
//            throughput *= exp(absorption * attenuationDistance);
//            attenuationDistance = 0.0f;
//        }

        if (!bsdfSample.delta) {
            float selectionPDF;
            constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
            LightSample lightSample = sampleLight(surfaceInteraction.position, light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);
            
            float3 wi = -ray.direction, wo = lightSample.wo;
            float3 pos1 = surfaceInteraction.position, pos2 = lightSample.position;
            float distance = lightSample.distance;
            
            float cosCamera = dot(wo, n);
            float cosLight = light.type == AREA_LIGHT ? dot(-wo, lightSample.normal) : 1.0f;
        
            if (cosCamera > 0.0f and cosLight > 0.0f and isVisible(pos1, surfaceInteraction.normal, pos2, lightSample.normal, instances, accelerationStructure)) {
                float3 BSDF = getBXDF(wi, wo, n, material);

                float G = cosCamera * cosLight / (distance * distance);
                float lightPDF = selectionPDF * lightSample.PDF;

                if (light.isDelta()) {
                    contribution += throughput * BSDF * lightSample.emission * G / lightPDF;
                } else {
                    float bsdfPDF = getPDF(wi, wo, n, material);
                    float weight = powerHeuristic(lightPDF, bsdfPDF);
                    contribution += throughput * BSDF * lightSample.emission * G * weight / lightPDF;
                }
            }
        }
                
        throughput *= bsdfSample.BSDF * abs(dot(wo, n)) / bsdfSample.PDF;
        
        if (isBlack(throughput))
            break;
        if (bounce > 4) {
            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
            if (sampler.r() > q) break;
            throughput /= q;
        }
        
        ray.origin = surfaceInteraction.position + wo * 1e-4f;
        ray.direction = wo;
        ray.min_distance = epsilon;
    }

    return contribution;
}
 
kernel void pathTracingKernel(device float3* accmulationBuffer,
                              constant Light* lights,
                              constant LightTriangle* lightTriangles,
                              constant int* instanceLightIndices,
                              constant AliasEntry* lightAliasEntries,
                              constant AliasEntry* lightTriangleAliasEntries,
                              constant AliasEntry* environmentMapAliasEntries,
                              constant Material* materials,
                              constant Textures* textures,
                              
                              texture2d<unsigned int> randomTex,
                              texture2d<float> environmentMapTexture,
                              
                              constant Uniforms& uniforms,
                              device MTLAccelerationStructureInstanceDescriptor* instances,
                              instance_acceleration_structure accelerationStructure,
                              
                              uint2 tid [[thread_position_in_grid]])
{
    unsigned int offset = randomTex.read(tid).x;
    Sampler sampler(offset, uniforms.frameIndex);
    
    float2 pixel = (float2) tid;
    pixel += sampler.r2() - 0.5f;
    
    if (pixel.x >= uniforms.width || pixel.y >= uniforms.height)
        return;

    float3 contribution = pathIntegrator(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, sampler, textures, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);
        
    accmulationBuffer[tid.y * uniforms.width + tid.x] += contribution;
}

kernel void finalizeAccumulationBDPT(uint2 tid [[thread_position_in_grid]],
                                    device float3* accumulation,
                                    device atomic_float* splatAccmulation,
                                    constant Uniforms& uniforms,
                                    texture2d<float, access::read_write> finalImage)
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    float3 contribution = accumulation[pixelIndex];
    float3 splatContribution = float3(atomic_load_explicit(&splatAccmulation[3 * pixelIndex + 0], memory_order_relaxed),
                                      atomic_load_explicit(&splatAccmulation[3 * pixelIndex + 1], memory_order_relaxed),
                                      atomic_load_explicit(&splatAccmulation[3 * pixelIndex + 2], memory_order_relaxed));
    
    float3 color = (contribution + splatContribution) / (uniforms.frameIndex + 1);
    finalImage.write(float4(color, 1.0f), tid);
}

kernel void finalizeAccumulation(uint2 tid [[thread_position_in_grid]],
                                device float3* accumulation,
                                constant Uniforms& uniforms,
                                texture2d<float, access::read_write> finalImage)
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    float3 contribution = accumulation[pixelIndex];
    
    float3 color = contribution / (uniforms.frameIndex + 1.0f);
    finalImage.write(float4(color, 1.0f), tid);
}
