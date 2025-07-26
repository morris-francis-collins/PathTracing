//
//  Integrators.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "Integrators.h"

using namespace metal;
using namespace raytracing;

float3 pathIntegrator(float2 pixel,
                      constant Uniforms& uniforms,
                      constant unsigned int& resourceStride,
                      device void *resources,
                      device MTLAccelerationStructureInstanceDescriptor *instances,
                      instance_acceleration_structure accelerationStructure,
                      device Light *lights,
                      device LightTriangle *lightTriangles,
                      device int *lightIndices,
                      texture2d<float> environmentMapTexture,
                      device float *environmentMapCDF,
                      array<texture2d<float>, MAX_TEXTURES> textureArray,
                      HaltonSampler haltonSampler
                      )
{
    constant Camera& camera = uniforms.camera;
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;

    ray ray;
    ray.origin = camera.position + camera.forward * 1e-4f;
    ray.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
    ray.min_distance = 1e-6f;
    ray.max_distance = INFINITY;

    float3 contribution = float3(0.0f);
    float3 throughput = float3(1.0f);
    float PDF = 1.0f;
    bool prevSpecular = true;
    
    bool inMedium = false;
    float attenuationDistance = 0.0f;
    
    for (int bounce = 0; bounce < MAX_PATH_LENGTH; bounce++) {
        IntersectionResult intersection = intersect(ray,
                                                    RAY_MASK_PRIMARY,
                                                    resources,
                                                    instances,
                                                    accelerationStructure,
                                                    false);
        
        if (intersection.type == intersection_type::none) {
            constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);
            float3 dir = ray.direction;

            float u = (atan2(dir.z, dir.x) + M_PI_F) / (2.0f * M_PI_F);
            float v = 1.0f - (asin(clamp(dir.y, -1.0f, 1.0f)) + M_PI_2_F) / M_PI_F;
            float4 textureValue = environmentMapTexture.sample(textureSampler, float2(u, v));

            if (prevSpecular) {
                contribution += ENVIRONMENT_MAP_SCALE * throughput * textureValue.xyz;
            } else {
                float lightPDF = environmentLightSamplePDF(float2(u, v), environmentMapCDF);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += ENVIRONMENT_MAP_SCALE * throughput * textureValue.xyz * weight;
            }
            
            break;
        }
        
        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, resources, instances, accelerationStructure, lightIndices, resourceStride, textureArray);
        Material material = surfaceInteraction.material;
        float3 n = surfaceInteraction.normal;
        
        if (surfaceInteraction.hitLight) {
            Light light = lights[surfaceInteraction.lightIndex];
            float3 color = light.color;

            if (prevSpecular) {
                contribution += throughput * color;
            } else {
                float lightPDF = getLightSelectionPDF(lights, uniforms, surfaceInteraction.lightIndex) * getLightSamplePDF(light);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += throughput * color * weight;
            }
            break;
        }
        
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, haltonSampler.r3());
        bsdfSample.BSDF *= surfaceInteraction.textureColor; // ensure to multiply by this
        
        PDF = bsdfSample.PDF;
        float3 wo = bsdfSample.wo;
        inMedium ^= bsdfSample.transmitted;
        float epsilon = calculateEpsilon(surfaceInteraction.position);
        prevSpecular = bsdfSample.delta;
        
        if (inMedium) {
            attenuationDistance += length(surfaceInteraction.position - ray.origin);
        } else if (bsdfSample.transmitted) {
            float3 absorption = log(material.color);
            throughput *= exp(absorption * attenuationDistance);
            attenuationDistance = 0.0f;
        }
                        
        if (!bsdfSample.delta) {
            float selectionPDF;
            Light light = selectLight(lights, lightTriangles, uniforms, haltonSampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(surfaceInteraction.position, light, lightTriangles, environmentMapTexture, environmentMapCDF, haltonSampler.r3());
            
            float3 wi = -ray.direction, wo = lightSample.wo;
            float3 pos1 = surfaceInteraction.position, pos2 = lightSample.position;
            float distance = lightSample.distance;
            
            float cosCamera = dot(wo, n);
            float cosLight = light.type == AREA_LIGHT ? dot(-wo, lightSample.normal) : 1.0f;
        
            if (cosCamera > 0.0f and cosLight > 0.0f and isVisible(pos1, pos2, resources, instances, accelerationStructure)) {
                float3 BSDF = getBXDF(wi, wo, n, material);
                BSDF *= surfaceInteraction.textureColor; // ensure

                float G = cosCamera * cosLight / (distance * distance);
                float lightPDF = selectionPDF * lightSample.PDF;

                if (light.delta) {
                    contribution += throughput * BSDF * lightSample.emission * G / lightPDF;
                } else {
                    float bsdfPDF = getPDF(wi, wo, n, material);
                    float weight = powerHeuristic(lightPDF, bsdfPDF);
                    contribution += throughput * BSDF * lightSample.emission * G * weight / lightPDF;
                }
            }
        }
        
        throughput *= bsdfSample.BSDF * abs(dot(wo, n)) / bsdfSample.PDF;
        if (all(throughput < 1e-10f)) break;
        
        if (bounce > 4) {
            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
            if (haltonSampler.r() > q) break;
            throughput /= q;
        }
        
        ray.origin = surfaceInteraction.position + calculateOffset(wo, n, epsilon);
        ray.direction = wo;
        ray.min_distance = epsilon;
    }

    return contribution;
}
