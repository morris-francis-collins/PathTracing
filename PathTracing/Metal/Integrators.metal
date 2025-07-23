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
                      device AreaLight *areaLights,
                      device LightTriangle *lightTriangles,
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
            break;
        }
        
        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, resources, instances, accelerationStructure, resourceStride, textureArray);
        Material material = surfaceInteraction.material;
        float3 n = surfaceInteraction.normal;
        
        if (surfaceInteraction.hitLight) {
            float3 color = 10 * float3(0.5f, 0.8f, 1.0f);
            color = float3(10.0f);

            if (prevSpecular) {
                contribution += throughput * color;
            } else {
                float lightPDF = 1 / (2 * 6.4);
                float weight = balanceHeuristic(PDF, lightPDF);
//                if (weight > 1.0f) debug(0.99);
//                if (weight < 0.0f) debug(-1232);
                if (throughput.x < 0.0f || throughput.y < 0.0f || throughput.z < 0.0f) debug(throughput);
//                weight = clamp(weight, 0.0f, 1.0f);
                contribution += throughput * color * weight;
            }
            break;
        }
        
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, haltonSampler.r3());
        bsdfSample.BSDF *= surfaceInteraction.textureColor;
        
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
        
        throughput *= bsdfSample.BSDF * abs(dot(wo, n)) / bsdfSample.PDF;
        if (all(throughput < 1e-10f)) break;
                
        if (!bsdfSample.delta) {
            float selectionPDF;
            AreaLight light = selectLight(areaLights, lightTriangles, uniforms, haltonSampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(light, lightTriangles, haltonSampler.r3());
            
            float3 wi = -ray.direction;
            float3 wo = lightSample.position - surfaceInteraction.position;
            float distance = length(wo);
            wo /= distance;
                        
            float cosCamera = dot(wo, n);
            float cosLight = dot(-wo, lightSample.normal);
                                    
            if (cosCamera > 0.0f and cosLight > 0.0f and isVisible(surfaceInteraction.position, lightSample.position, n, lightSample.normal, resources, instances, accelerationStructure)) {
                
                float3 BSDF = getBXDF(wi, wo, n, material);
                float bsdfPDF = getPDF(wi, wo, n, material);
                
                if (material.metallic > 0.0f) {
                    BSDF = conductorBSDF(wi, wo, n, material);
                    bsdfPDF = conductorPDF(wi, wo, n, material);
//                    DEBUG("metal, bsdf: %f, pdf: %f", BSDF.x, bsdfPDF);
//                    if (any(BSDF < 0.0f)) DEBUG("metal BSDF < 0");
//                    if (bsdfPDF < 0.0f) DEBUG("metal PDF < 0");

                } else if (material.BXDFs & SPECULAR_TRANSMISSION) {
                    BSDF = dielectricBSDF(wi, wo, n, material);
                    bsdfPDF = dielectricPDF(wi, wo, n, material);
//                    DEBUG("glass, bsdf: %f, pdf: %f", BSDF.x, bsdfPDF);
                    if (any(BSDF < 0.0f)) DEBUG("glass BSDF < 0");
                    if (bsdfPDF < 0.0f) DEBUG("glass PDF < 0");

                } else {
                    BSDF = diffuseBRDF(material);
                    bsdfPDF = diffusePDF(wi, wo, n);
//                    DEBUG("diffuse, bsdf: %f, pdf: %f", BSDF.x, bsdfPDF);
                }
                
                BSDF = max(BSDF, float3(0.0f));
//                PDF = clamp(PDF, 0.0f, 1.0f);
                

                float lightPDF =  1 / (2 * 6.4);

                float weight = balanceHeuristic(lightPDF, bsdfPDF);
                float G = cosCamera * cosLight / (distance * distance);
//                debug(lightSample.emission);
                
                if (bsdfPDF > 0.0f)
                    contribution += throughput * BSDF * lightSample.emission * G * weight / lightPDF;
            }
        }
        
        float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
        if (haltonSampler.r() > q) break;
        throughput /= q;
        
//        throughput = abs(throughput);
//        
//        bsdfSample.PDF = clamp(bsdfSample.PDF, 0.0f, 1.0f);
        
//        if (bsdfSample.PDF < 1e-6f) {
//            DEBUG("pdf near 0: %f, log: %f", bsdfSample.PDF, log10(bsdfSample.PDF));
////            break;
//        }
        
        
//        if (any(bsdfSample.BSDF < 0.0f)) DEBUG("bsdfSample.BSDF < 0");
//        if (bsdfSample.PDF < 0.0f) DEBUG("bsdfSample.PDF < 0");

//        if (any(throughput < 0.0f)) DEBUG("throughput < 0");



        ray.origin = surfaceInteraction.position + calculateOffset(wo, n, epsilon);
        ray.direction = wo;
        ray.min_distance = epsilon;
    }

    return contribution;
}
