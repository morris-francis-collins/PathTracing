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
                      constant Light *lights,
                      constant LightTriangle *lightTriangles,
                      constant int *lightIndices,
                      texture2d<float> environmentMapTexture,
                      constant float *environmentMapCDF,
                      array<texture2d<float>, MAX_TEXTURES> textureArray,
                      thread Sampler& sampler
                      )
{
    ray ray = generateRay(pixel, uniforms);

    float3 contribution = float3(0.0f);
    float3 throughput = float3(1.0f);
    float PDF = 0.0f;
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
            float2 uv = getEnvironmentMapUV(ray.direction);
            float3 emission = environmentMapEmission(uv, environmentMapTexture);
            
            if (all(emission < 1e-10f))
                break;

            if (prevSpecular) {
                contribution += throughput * emission;
            } else {
                float lightPDF = environmentLightSamplePDF(uv, environmentMapCDF);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += throughput * emission * weight;
            }

            break;
        }
        
        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, resources, instances, accelerationStructure, lightIndices, resourceStride, textureArray);
        Material material = surfaceInteraction.material;
        
        float3 n = surfaceInteraction.normal;

        if (surfaceInteraction.hitLight) {
            constant Light& light = lights[surfaceInteraction.lightIndex];
            float3 color = light.color;

            if (prevSpecular) {
                contribution += throughput * color;
            } else {
                float lightPDF = getLightSelectionPDF(light, lights, uniforms) * getLightSamplePDF(light);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += throughput * color * weight;
            }

            break;
        }
        
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, sampler.r3());
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
            constant Light& light = selectLight(lights, lightTriangles, uniforms, sampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(surfaceInteraction.position, light, lightTriangles, environmentMapTexture, environmentMapCDF, sampler.r3());
            
            float3 wi = -ray.direction, wo = lightSample.wo;
            float3 pos1 = surfaceInteraction.position, pos2 = lightSample.position;
            float distance = lightSample.distance;
            
            float cosCamera = dot(wo, n);
            float cosLight = light.type == AREA_LIGHT ? dot(-wo, lightSample.normal) : 1.0f;
        
            if (cosCamera > 0.0f and cosLight > 0.0f and isVisible(pos1, surfaceInteraction.normal, pos2, lightSample.normal, resources, instances, accelerationStructure)) {
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
        
        if (isBlack(throughput))
            break;
        if (bounce > 4) {
            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
            if (sampler.r() > q) break;
            throughput /= q;
        }
        
        ray.origin = surfaceInteraction.position + calculateOffset(wo, n, epsilon);
        ray.direction = wo;
        ray.min_distance = epsilon;
    }

    return contribution;
}

int tracePath(float2 pixel,
              constant Uniforms& uniforms,
              constant unsigned int& resourceStride,
              device void *resources,
              device MTLAccelerationStructureInstanceDescriptor *instances,
              instance_acceleration_structure accelerationStructure,
              constant Light *lights,
              constant LightTriangle *lightTriangles,
              constant int *lightIndices,
              texture2d<float> environmentMapTexture,
              constant float *environmentMapCDF,
              array<texture2d<float>, MAX_TEXTURES> textureArray,
              thread Sampler& sampler,
              ray ray,
              int maxDepth,
              thread PathVertex *vertices,
              VertexType type,
              float3 throughput,
              float forwardPDF
              )
{
    int bounces = 1;
    
    for (int bounce = 1; bounce < maxDepth; bounce++) {
        thread PathVertex& vx = vertices[bounces];
        thread PathVertex& prev = vertices[bounces - 1];
        
        IntersectionResult intersection = intersect(ray,
                                                    RAY_MASK_PRIMARY,
                                                    resources,
                                                    instances,
                                                    accelerationStructure,
                                                    false);
        
        if (intersection.type == intersection_type::none) {
            if (type == CAMERA_VERTEX) {
                float3 endPos = ray.origin + ray.direction * 4 * SCENE_RADIUS;
                vx = createLightVertex(nullptr, endPos, float3(0.0f), throughput, forwardPDF);
                bounces++;
            }
            break;
        }

        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, resources, instances, accelerationStructure, lightIndices, resourceStride, textureArray);
        Material material = surfaceInteraction.material;
        float3 n = surfaceInteraction.normal;
        
        vx = createSurfaceVertex(surfaceInteraction, throughput, forwardPDF, prev);

        if (++bounces >= maxDepth) {
            break;
        }

        if (surfaceInteraction.hitLight) { // scatter from lights?
            break;
        }
                
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, sampler.r3());
        bsdfSample.BSDF *= surfaceInteraction.textureColor; // ensure to multiply by this
        
        float3 wo = bsdfSample.wo;
        float epsilon = calculateEpsilon(surfaceInteraction.position);
                
        forwardPDF = bsdfSample.PDF;
        throughput *= bsdfSample.BSDF * abs(dot(wo, n)) / bsdfSample.PDF;
                
        float reversePDF = getPDF(wo, -ray.direction, n, material);
        
        if (bsdfSample.delta) {
            vx.delta = true;
            forwardPDF = 0.0f;
            reversePDF = 0.0f;
        }
        
        prev.reversePDF = vx.convertDensity(reversePDF, prev);

        if (isBlack(throughput))
            break;
        if (bounce > 4) {
            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
            if (sampler.r() > q) break;
            throughput /= q;
        }

        ray.origin = surfaceInteraction.position + calculateOffset(wo, n, epsilon);
        ray.direction = wo;
        ray.min_distance = epsilon;
    }

    return bounces;
}

int traceCameraPath(float2 pixel,
                    constant Uniforms& uniforms,
                    constant unsigned int& resourceStride,
                    device void *resources,
                    device MTLAccelerationStructureInstanceDescriptor *instances,
                    instance_acceleration_structure accelerationStructure,
                    constant Light *lights,
                    constant LightTriangle *lightTriangles,
                    constant int *lightIndices,
                    texture2d<float> environmentMapTexture,
                    constant float *environmentMapCDF,
                    array<texture2d<float>, MAX_TEXTURES> textureArray,
                    thread Sampler& sampler,
                    thread PathVertex *cameraVertices
                    )
{
    constant Camera& camera = uniforms.camera;
    ray ray = generateRay(pixel, uniforms);
    
    float positionPDF, directionPDF;
//    cameraRayPDF(camera, ray.direction, positionPDF, directionPDF); // better results when directionPDF = 0? possibly due to pinhole
    positionPDF = 1.0f, directionPDF = 0.0f;
    
    cameraVertices[0] = createCameraVertex(&camera, camera.position, camera.forward, float3(1.0f));
    cameraVertices[0].forwardPDF = positionPDF;
    float3 throughput = float3(1.0f);

    return tracePath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler, ray, MAX_CAMERA_PATH_LENGTH, cameraVertices, CAMERA_VERTEX, throughput, directionPDF);
}

int traceLightPath(float2 pixel,
                   constant Uniforms& uniforms,
                   constant unsigned int& resourceStride,
                   device void *resources,
                   device MTLAccelerationStructureInstanceDescriptor *instances,
                   instance_acceleration_structure accelerationStructure,
                   constant Light *lights,
                   constant LightTriangle *lightTriangles,
                   constant int *lightIndices,
                   texture2d<float> environmentMapTexture,
                   constant float *environmentMapCDF,
                   array<texture2d<float>, MAX_TEXTURES> textureArray,
                   thread Sampler& sampler,
                   thread PathVertex *lightVertices
                   )
{
    float selectionPDF;
    constant Light& light = selectLight(lights, lightTriangles, uniforms, sampler.r(), selectionPDF);
    LightEmissionSample lightEmissionSample = sampleLightEmission(light, lightTriangles, environmentMapTexture, environmentMapCDF, sampler.r2(), sampler.r3());
    float positionPDF = lightEmissionSample.positionPDF;
    float directionPDF = lightEmissionSample.directionPDF;
    float3 normal = lightEmissionSample.normal;

    ray ray;
    float epsilon = calculateEpsilon(lightEmissionSample.position);
    ray.origin = lightEmissionSample.position + calculateOffset(lightEmissionSample.wo, normal, epsilon);
    ray.direction = lightEmissionSample.wo;
    ray.min_distance = epsilon;
    ray.max_distance = INFINITY;

    lightVertices[0] = createLightVertex(&light, lightEmissionSample.position, normal, light.color, positionPDF * selectionPDF);
    float3 throughput = light.color / (selectionPDF * positionPDF * directionPDF);
    if (light.type == AREA_LIGHT)
        throughput *= abs(dot(lightVertices[0].normal(), ray.direction));

    int numVertices = tracePath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler, ray, MAX_LIGHT_PATH_LENGTH, lightVertices, LIGHT_VERTEX, throughput, directionPDF);

    if (lightVertices[0].isInfiniteLight()) {
        if (numVertices > 0) {
            lightVertices[1].forwardPDF = positionPDF;
            if (lightVertices[1].isOnSurface())
                lightVertices[1].forwardPDF *= abs(dot(ray.direction, lightVertices[1].normal()));
        }
        lightVertices[0].forwardPDF = infiniteLightDensity(ray.direction, lights, uniforms, environmentMapCDF);
    }
    
    return numVertices;
}

float3 calculateGeometricTerm(thread PathVertex& cameraVertex,
                              thread PathVertex& lightVertex,
                              device void *resources,
                              device MTLAccelerationStructureInstanceDescriptor *instances,
                              instance_acceleration_structure accelerationStructure
                              )
{
    float3 connectionVector = lightVertex.position() - cameraVertex.position();
    float connectionDistance = length(connectionVector);
    float3 connectionDirection = connectionVector / connectionDistance;
                
    float G = (lightVertex.isInfiniteLight() || lightVertex.ei.light->type == DIRECTIONAL_LIGHT) ? 1.0f : 1.0f / max(connectionDistance * connectionDistance, 0.0f);

    if (cameraVertex.isOnSurface()) {
        float cosCamera = dot(cameraVertex.normal(), connectionDirection);
        if (cosCamera < 0.0f)
            return float3(0.0f);
        G *= abs(cosCamera);
    }
    
    if (lightVertex.isOnSurface()) {
        float cosLight = dot(lightVertex.normal(), -connectionDirection);
        if (cosLight < 0.0f)
            return float3(0.0f);
        G *= abs(cosLight);
    }
    
    if (!isVisible(cameraVertex.position(), cameraVertex.normal(), lightVertex.position(), lightVertex.normal(), resources, instances, accelerationStructure)) {
        return float3(0.0f);
    }

    return G;
}

float calculateMISWeight(constant Uniforms& uniforms,
                         constant Light* lights,
                         constant float *environmentMapCDF,
                         thread PathVertex *cameraVertices,
                         thread PathVertex *lightVertices,
                         int c, int l,
                         thread PathVertex& sampled
                         )
{
    if (c + l == 2) return 1.0f;
    auto remap0 = [&](float x) -> float { return x != 0.0f ? x : 1.0f; };

    int ci = c - 1;
    int cip = ci - 1;
    int li = l - 1;
    int lip = li - 1;

    PathVertex origVx;
    
    if (l == 1) {
        origVx = lightVertices[0];
        lightVertices[0] = sampled;
    }

    float originalCameraReverse = cameraVertices[ci].reversePDF;
    bool  origCamDelta   = cameraVertices[ci].delta;
    float origCamPrevRev = (cip >= 0) ? cameraVertices[cip].reversePDF : 0.0f;
    bool  origCamPrevDel = (cip >= 0) ? cameraVertices[cip].delta   : false;

    float origLgtRev     = lightVertices[li].reversePDF;
    bool  origLgtDelta   = lightVertices[li].delta;
    float origLgtPrevRev = (lip >= 0) ? lightVertices[lip].reversePDF : 0.0f;
    bool  origLgtPrevDel = (lip >= 0) ? lightVertices[lip].delta   : false;
                
    if (ci >= 0) {
        if (li >= 0) {
            cameraVertices[ci].reversePDF = lightVertices[li].PDF(lightVertices[lip], cameraVertices[ci], environmentMapCDF);
        } else { // l = 0 case
            constant Light& light = lights[cameraVertices[ci].si.lightIndex]; // originally origin
            cameraVertices[ci].reversePDF = cameraVertices[ci].lightOriginPDF(light, lights, uniforms, cameraVertices[cip], environmentMapCDF);
        }
        cameraVertices[ci].delta = false;
    }
    
    if (cip >= 0) {
        if (li >= 0) {
            cameraVertices[cip].reversePDF = cameraVertices[ci].PDF(lightVertices[li], cameraVertices[cip], environmentMapCDF);
        } else { // l = 0 case
            constant Light& light = lights[cameraVertices[ci].si.lightIndex]; // originally direction
            cameraVertices[cip].reversePDF = cameraVertices[ci].lightDirectionPDF(light, cameraVertices[cip], environmentMapCDF);
        }
    }
    
    if (li >= 0) {
        lightVertices[li].reversePDF = cameraVertices[ci].PDF(cameraVertices[cip], lightVertices[li], environmentMapCDF);
        lightVertices[li].delta = false;
    }
    
    if (lip >= 0) {
        lightVertices[lip].reversePDF = lightVertices[li].PDF(cameraVertices[ci], lightVertices[lip], environmentMapCDF);
    }
    
    float sum = 0.0f;
    float r = 1.0f;

    for (int i = ci; i > 0; i--) {
        r *= remap0(cameraVertices[i].reversePDF) / remap0(cameraVertices[i].forwardPDF);

        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sum += r;
    }

    r = 1.0f;
    
    for (int i = li; i >= 0; i--) {
        r *= remap0(lightVertices[i].reversePDF) / remap0(lightVertices[i].forwardPDF);
        bool prevDelta = (i > 0) ? lightVertices[i - 1].delta : lightVertices[0].isDeltaLight();

        if (!lightVertices[i].delta && !prevDelta)
            sum += r;
    }
        
    cameraVertices[ci].reversePDF = originalCameraReverse;
    cameraVertices[ci].delta   = origCamDelta;
    if (cip >= 0) {
        cameraVertices[cip].reversePDF = origCamPrevRev;
        cameraVertices[cip].delta   = origCamPrevDel;
    }

    lightVertices[li].reversePDF = origLgtRev;
    lightVertices[li].delta = origLgtDelta;
    if (lip >= 0) {
        lightVertices[lip].reversePDF = origLgtPrevRev;
        lightVertices[lip].delta   = origLgtPrevDel;
    }
    
    if (l == 1) {
        lightVertices[0] = origVx;
    }
        
    return 1.0f / (1.0f + sum);
}

uint2 projectToScreen(float3 worldPos, constant Uniforms& uniforms)
{
    float3 toPoint = worldPos - uniforms.camera.position;
    float zCam = dot(toPoint, uniforms.camera.forward);
    if (zCam <= 0.0f)
        return uint2(UINT_MAX, UINT_MAX);

    float3 normalizedRight = normalize(uniforms.camera.right);
    float3 normalizedUp = normalize(uniforms.camera.up);

    float fieldOfView = CAMERA_FOV_ANGLE * (M_PI_F / 180.0f);
    float imagePlaneHeight = tan(fieldOfView / 2.0f);
    float imagePlaneWidth = imagePlaneHeight * float(uniforms.width) / float(uniforms.height);

    float xProj = dot(toPoint, normalizedRight) / (zCam * imagePlaneWidth);
    float yProj = dot(toPoint, normalizedUp) / (zCam * imagePlaneHeight);
    
    float2 uv;
    uv.x = 0.5f + 0.5f * xProj;
    uv.y = 0.5f + 0.5f * yProj;
    
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return uint2(UINT_MAX, UINT_MAX);
    }

    uint px = min(uint(uv.x * float(uniforms.width)), uniforms.width - 1);
    uint py = min(uint(uv.y * float(uniforms.height)), uniforms.height - 1);
    return uint2(px, py);
}

void splat(texture2d<float, access::read_write> splatTex,
           constant Uniforms& uniforms,
           uint2 pixelCoordinate,
           float3 color,
           device atomic_float* splatBuffer
           )
{
    if (pixelCoordinate.x >= uniforms.width || pixelCoordinate.y >= uniforms.height) {
        return;
    }
    
    uint width = uniforms.width;
    uint pixelIndex = (pixelCoordinate.y * width + pixelCoordinate.x) * 3;
        
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 0], color.r, memory_order_relaxed);
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 1], color.g, memory_order_relaxed);
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 2], color.b, memory_order_relaxed);
}

float3 connectVertices(constant Uniforms& uniforms,
                       constant unsigned int& resourceStride,
                       device void *resources,
                       device MTLAccelerationStructureInstanceDescriptor *instances,
                       instance_acceleration_structure accelerationStructure,
                       constant Light *lights,
                       constant LightTriangle *lightTriangles,
                       constant int *lightIndices,
                       texture2d<float> environmentMapTexture,
                       constant float *environmentMapCDF,
                       array<texture2d<float>, MAX_TEXTURES> textureArray,
                       thread Sampler& sampler,
                       thread PathVertex *cameraVertices,
                       thread PathVertex *lightVertices,
                       int c, int l
                       )
{
    if (c > 1 && l != 0 && cameraVertices[c - 1].type == LIGHT_VERTEX)
        return float3(0.0f);
    
    float3 contribution = float3(0.0f);
    PathVertex sampled;
    
    if (l == 0) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
//        return contribution;
        if (cameraVertex.isLight()) {
            contribution = cameraVertex.getLightEmission(cameraVertices[c - 2], lights, environmentMapTexture) * cameraVertex.throughput;
        }
    } else if (c == 1) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        thread PathVertex& lightVertex = lightVertices[l - 1];
//        return contribution;
        if (lightVertex.isConnectible()) {
            float3 We = cameraWe(uniforms.camera, lightVertex.position());
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex) * lightVertex.si.textureColor;
            contribution = We * lightVertex.throughput * lightBSDF;
            
            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, resources, instances, accelerationStructure);
        }
    } else if (l == 1) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
//        return contribution;
        if (cameraVertex.isConnectible()) {
            float selectionPDF;
            constant Light& light = selectLight(lights, lightTriangles, uniforms, sampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(cameraVertex.position(), light, lightTriangles, environmentMapTexture, environmentMapCDF, sampler.r3());
            sampled = createLightVertex(&light, lightSample.position, lightSample.normal, lightSample.emission / (selectionPDF * lightSample.PDF), selectionPDF * lightSample.PDF);

            contribution = cameraVertex.throughput * cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), sampled) * sampled.throughput * cameraVertex.si.textureColor;
                        
            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, sampled, resources, instances, accelerationStructure);
        }
    } else {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        thread PathVertex& lightVertex = lightVertices[l - 1];
//        return contribution;
        if (cameraVertex.isConnectible() && lightVertex.isConnectible()) {
            float3 cameraBSDF = cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), lightVertex) * cameraVertex.si.textureColor;
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex) * lightVertex.si.textureColor;
            contribution = cameraVertex.throughput * lightVertex.throughput * cameraBSDF * lightBSDF;

            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, resources, instances, accelerationStructure);
        }
    }
    
    float MISWeight = !isBlack(contribution) ? calculateMISWeight(uniforms, lights, environmentMapCDF, cameraVertices, lightVertices, c, l, sampled) : 0.0f;
    contribution *= MISWeight;

    return contribution;
}

float3 bidirectionalPathIntegrator(float2 pixel,
                                   constant Uniforms& uniforms,
                                   constant unsigned int& resourceStride,
                                   device void *resources,
                                   device MTLAccelerationStructureInstanceDescriptor *instances,
                                   instance_acceleration_structure accelerationStructure,
                                   constant Light *lights,
                                   constant LightTriangle *lightTriangles,
                                   constant int *lightIndices,
                                   texture2d<float> environmentMapTexture,
                                   constant float *environmentMapCDF,
                                   array<texture2d<float>, MAX_TEXTURES> textureArray,
                                   thread Sampler& sampler,
                                   texture2d<float, access::read_write> splatTex,
                                   device atomic_float* splatBuffer
                                   )
{
    PathVertex cameraVertices[MAX_CAMERA_PATH_LENGTH];
    PathVertex lightVertices[MAX_LIGHT_PATH_LENGTH];
    
    int cameraPathLength = traceCameraPath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler, cameraVertices);
    
    int lightPathLength = traceLightPath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler, lightVertices);

    float3 totalContribution = float3(0.0f);
    
    for (int c = 1; c <= cameraPathLength; c++) {
        for (int l = 0; l <= lightPathLength; l++) {
            int depth = c + l - 2;
            if ((c == 1 && l == 1) || depth < 0 || depth > MAX_PATH_LENGTH)
                continue;

            float3 contribution = connectVertices(uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler, cameraVertices, lightVertices, c, l);
            
            if (any(contribution < 0.0f) || any(isnan(contribution)) || any(isinf(contribution))) {
//                DEBUG("Invalid contribution - c: %d, l: %d, float3(%f, %f, %f)", c, l, contribution.x, contribution.y, contribution.z);
                continue;
            }

            if (c == 1) {
                uint2 pixelCoord = projectToScreen(lightVertices[l - 1].position(), uniforms);
                splat(splatTex, uniforms, pixelCoord, contribution, splatBuffer);
            } else {
                totalContribution += contribution;
            }
        }
    }
    
    return totalContribution;
}
