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

float3 sampleNEE(float3 throughput,
                 ray ray,
                        
                 constant Light* lights,
                 constant LightTriangle* lightTriangles,
                 thread Sampler& sampler,
                 constant Textures* textures,
                 constant AliasEntry* lightAliasEntries,
                 constant AliasEntry* lightTriangleAliasEntries,
                 texture2d<float> environmentMapTexture,
                 constant AliasEntry* environmentMapAliasEntries,
                        
                 thread SurfaceInteraction& si,

                 constant Uniforms& uniforms,
                 device MTLAccelerationStructureInstanceDescriptor* instances,
                 instance_acceleration_structure accelerationStructure)
{
    float selectionPDF;
    constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
    LightSample ls = sampleLight(si.position, light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);

    float3 wi = -ray.direction, wo = ls.wo;
    float cosCamera = dot(wo, si.normal);
    float cosLight = light.type == AREA_LIGHT ? dot(-wo, ls.normal) : 1.0f;

    if (cosCamera > 0.0f && cosLight > 0.0f && isVisible(si.position, si.normal, ls.position, ls.normal, instances, accelerationStructure)) {
        float3 BSDF = getBXDF(wi, wo, si.normal, si.material, Radiance);
        float G = cosCamera * cosLight / (ls.distance * ls.distance);
        float lightPDF = selectionPDF * ls.PDF;

        if (light.isDelta()) {
            return throughput * BSDF * ls.emission * G / lightPDF;
        } else {
            float bsdfPDF = getPDF(wi, wo, si.normal, si.material);
            float weight = powerHeuristic(lightPDF, bsdfPDF);
            return throughput * BSDF * ls.emission * G * weight / lightPDF;
        }
    }
    
    return float3(0.0f);
}

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

        if (surfaceInteraction.material.BXDFs == DIFFUSE || surfaceInteraction.material.BXDFs == CONDUCTOR || surfaceInteraction.material.BXDFs == DIELECTRIC_REFLECTION) { // TODO: move to surface interaction
            if (dot(-ray.direction, surfaceInteraction.normal) < 0.0f)
                surfaceInteraction.normal = -surfaceInteraction.normal;
        }

        if (surfaceInteraction.material.alphaMode == ALPHA_MASK && surfaceInteraction.material.alpha < surfaceInteraction.material.alphaCutoff) {
            ray.origin = surfaceInteraction.position + ray.direction * 1e-4f;
            continue;
        }

        if (surfaceInteraction.material.alphaMode == ALPHA_BLEND && sampler.r() > surfaceInteraction.material.alpha) {
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
        
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, surfaceInteraction.material, Radiance, sampler.r3());
        
        PDF = bsdfSample.PDF;
        float3 wo = bsdfSample.wo;
        float epsilon = calculateEpsilon(surfaceInteraction.position);
        prevSpecular = bsdfSample.delta;

        if (!bsdfSample.delta) {
            contribution += sampleNEE(throughput, ray, lights, lightTriangles, sampler, textures, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries, surfaceInteraction, uniforms, instances, accelerationStructure);
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
 
kernel void pathTracingKernel(device float* accumulation,
                              
                              constant Light* lights,
                              constant LightTriangle* lightTriangles,
                              constant int* instanceLightIndices,
                              constant AliasEntry* lightAliasEntries,
                              constant AliasEntry* lightTriangleAliasEntries,
                              constant AliasEntry* environmentMapAliasEntries,
                              constant Material* materials,
                              constant Textures* textures,
                              
                              constant uint* sobolValues,
                              
                              texture2d<unsigned int> randomTex,
                              texture2d<float> environmentMapTexture,
                              
                              constant Uniforms& uniforms,
                              device MTLAccelerationStructureInstanceDescriptor* instances,
                              instance_acceleration_structure accelerationStructure,
                              
                              uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    unsigned int offset = randomTex.read(tid).x;
    Sampler sampler(offset, uniforms.frameIndex, sobolValues);
    
    float2 pixel = (float2) tid;
    pixel += sampler.r2() - 0.5f;
    
    float3 contribution = pathIntegrator(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, sampler, textures, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    
    if (uniforms.frameIndex == 0u)
        setContribution(float3(0.0f), pixelIndex, accumulation);
    addContribution(contribution, pixelIndex, accumulation);
}

kernel void finalizeAccumulation(device float* accumulation,
                                 constant Uniforms& uniforms,
                                 texture2d<float, access::read_write> finalImage,
                                 
                                 uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    float3 contribution = getFloat3FromAccumulation(pixelIndex, accumulation);
    
    float3 color = contribution / (uniforms.frameIndex + 1.0f);
    finalImage.write(float4(color, 1.0f), tid);
}
