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
                      thread HaltonSampler& haltonSampler
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
            float3 textureValue = environmentMapTexture.sample(textureSampler, float2(u, v)).xyz;
            
            if (all(textureValue < 1e-10f)) break;

            if (prevSpecular) {
                contribution += ENVIRONMENT_MAP_SCALE * throughput * textureValue;
            } else {
                float lightPDF = environmentLightSamplePDF(float2(u, v), environmentMapCDF);
                float weight = powerHeuristic(PDF, lightPDF);
                contribution += ENVIRONMENT_MAP_SCALE * throughput * textureValue * weight;
            }

            break;
        }
        
        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, resources, instances, accelerationStructure, lightIndices, resourceStride, textureArray);
        Material material = surfaceInteraction.material;
        float3 n = surfaceInteraction.normal;
        
        if (surfaceInteraction.hitLight) {
//            break;
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
            constant Light& light = selectLight(lights, lightTriangles, uniforms, haltonSampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(surfaceInteraction.position, light, lightTriangles, environmentMapTexture, environmentMapCDF, haltonSampler.r3());
            
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
              thread HaltonSampler& haltonSampler,
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
        if (all(throughput < 1e-10f))
            break;
        
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
                // create light vx. we have no environment lights now so this is a placeholder.
//                bounces++;
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

        if (surfaceInteraction.hitLight) {
            break;
        }

                
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, haltonSampler.r3());
        bsdfSample.BSDF *= surfaceInteraction.textureColor; // ensure to multiply by this
        
        float3 wo = bsdfSample.wo;
        float epsilon = calculateEpsilon(surfaceInteraction.position);
                
        forwardPDF = bsdfSample.PDF;
        throughput *= bsdfSample.BSDF * abs(dot(wo, n)) / bsdfSample.PDF;
                
        float reversePDF = getPDF(wo, -ray.direction, n, material);
        
        if (bsdfSample.delta) {
            vx.delta = true;
            vx.forwardPDF = 0.0f;
            vx.reversePDF = 0.0f;
        }
        
        prev.reversePDF = convertDensity(reversePDF, vx, prev);
//        if (type == CAMERA_VERTEX && bounces == 1) {
//            DEBUG("camera surface?: %d, rev pdf: %f", prev.isOnSurface(), prev.reversePDF);
//        }

//        if (bounce > 4) {
//            float q = clamp(calculateLuminance(throughput), 0.05f, 1.0f);
//            if (haltonSampler.r() > q) break;
//            throughput /= q;
//        }

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
                    thread HaltonSampler& haltonSampler,
                    thread PathVertex *cameraVertices
                    )
{
    constant Camera& camera = uniforms.camera;
    ray ray = generateRay(pixel, uniforms);
    
    float positionPDF, directionPDF;
    cameraRayPDF(camera, ray.direction, positionPDF, directionPDF);
    
    cameraVertices[0] = createCameraVertex(camera, camera.position, camera.forward, float3(1.0f));
    cameraVertices[0].forwardPDF = positionPDF;
    float3 throughput = float3(1.0f);
//    if (pixel.x < 405.0f && pixel.x > 395.0f && pixel.y > 295.0f && pixel.y < 305.0f)
//    if (dot(camera.forward, ray.direction) >= 0.95f)
//        DEBUG("pixel xy: %f %f, cos: %f, dirpdf: %f", pixel.x, pixel.y, dot(camera.forward, ray.direction), directionPDF);

    return tracePath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, haltonSampler, ray, MAX_CAMERA_PATH_LENGTH, cameraVertices, CAMERA_VERTEX, throughput, directionPDF);
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
                   thread HaltonSampler& haltonSampler,
                   thread PathVertex *lightVertices
                   )
{
    float selectionPDF;
    constant Light& light = selectLight(lights, lightTriangles, uniforms, haltonSampler.r(), selectionPDF);
    LightEmissionSample lightEmissionSample = sampleLightEmission(light, lightTriangles, environmentMapTexture, environmentMapCDF, haltonSampler.r2(), haltonSampler.r3());
    float positionPDF = lightEmissionSample.positionPDF;
    float directionPDF = lightEmissionSample.directionPDF;
    float3 normal = lightEmissionSample.normal;

    ray ray;
    float epsilon = calculateEpsilon(lightEmissionSample.position);
    ray.origin = lightEmissionSample.position + calculateOffset(lightEmissionSample.wo, normal, epsilon);
    ray.direction = lightEmissionSample.wo;
    ray.min_distance = epsilon;
    ray.max_distance = INFINITY;
    
    lightVertices[0] = createLightVertex(light, ray.origin, normal, light.color, positionPDF * selectionPDF);
    float3 throughput = light.color / (selectionPDF * positionPDF * directionPDF);
//    DEBUG("light throughput: %f", length(throughput));
//    DEBUG("selection: %f, position: %f, direction %f:", selectionPDF, positionPDF, directionPDF);
    if (lightVertices[0].isOnSurface()) throughput *= abs(dot(lightVertices[0].normal(), ray.direction));
    
    return tracePath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, haltonSampler, ray, MAX_LIGHT_PATH_LENGTH, lightVertices, LIGHT_VERTEX, throughput, directionPDF);
}

float3 calculateGeometricTerm(thread PathVertex& cameraVertex,
                              thread PathVertex& lightVertex,
                              device void *resources,
                              device MTLAccelerationStructureInstanceDescriptor *instances,
                              instance_acceleration_structure accelerationStructure
                              )
{
    if (!isVisible(cameraVertex.position(), cameraVertex.normal(), lightVertex.position(), lightVertex.normal(), resources, instances, accelerationStructure)) {
        return float3(0.0f);
    }
//    debug(lightVertex.position(), cameraVertex.position());
    float3 connectionVector = lightVertex.position() - cameraVertex.position();
    float connectionDistance = length(connectionVector);
    float3 connectionDirection = connectionVector / connectionDistance;
    
    float G = 1.0f / (connectionDistance * connectionDistance);
    //    if (length(lightVertex.normal()) < 0.1) {
    //        DEBUG("no light normal");
    //    }
    //    debug(lightVertex.normal());
    //    if (cosCamera > 0.1) {
    //        debug(cosCamera);
    //    }
    //    DEBUG("normal: float3(%f, %f, %f), connectiondir: float3(%f, %f, %f)",
    //          cameraVertex.normal().x, cameraVertex.normal().y, cameraVertex.normal().z, connectionDirection.x, connectionDirection.y, connectionDirection.z);
    //    DEBUG("pos1: float3(%f, %f, %f), pos2: float3(%f, %f, %f)",
    //          cameraVertex.position().x, cameraVertex.position().y, cameraVertex.position().z, lightVertex.position().x, lightVertex.position().y, lightVertex.position().z);
    if (cameraVertex.isOnSurface()) {
        
//        DEBUG("coscamera: %f", dot(cameraVertex.normal(), connectionDirection));
        G *= (dot(cameraVertex.normal(), connectionDirection));
    }
//    
    if (lightVertex.isOnSurface()) {
//        DEBUG("coslight: %f", dot(lightVertex.normal(), -connectionDirection));
        G *= (dot(lightVertex.normal(), -connectionDirection));
    }
    
    return G;
}

float calculateMISWeight(constant Uniforms& uniforms,
                         constant Light* lights,
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
//    if (c == 1) {
//        origVx = cameraVertices[0];
//        cameraVertices[0] = sampled;
//    }
    
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
//            if (lip < 0) DEBUG("lip < 0, li type: %d", lightVertices[li].type);
            cameraVertices[ci].reversePDF = lightVertices[li].PDF(lightVertices[lip], cameraVertices[ci]);
//            cameraVertices[ci].reversePDF = evaluatePDF(lightVertices[li], lightVertices[lip], cameraVertices[ci]);
//            debug(lightVertices[li].type == LIGHT_VERTEX);
//            DEBUG("ci . %f", (cameraVertices[ci].reversePDF));
//            if (cameraVertices[ci].reversePDF < 0)
//                DEBUG("ci li >= 0 negative. %f", cameraVertices[ci].reversePDF);
        } else {
            constant Light& light = lights[cameraVertices[ci].si.lightIndex];
            cameraVertices[ci].reversePDF = cameraVertices[ci].lightOriginPDF(light, lights, uniforms);
//            DEBUG("ci else. %f", cameraVertices[ci].reversePDF);

//            if (cameraVertices[ci].reversePDF < 0)
//                DEBUG("ci else negative. %f", cameraVertices[ci].reversePDF);

        }
        cameraVertices[ci].delta = false;
    }
    
    if (cip >= 0) {
//        DEBUG("ci: c: %d, l: %d", c, l);
        if (li >= 0) {
            cameraVertices[cip].reversePDF = cameraVertices[ci].PDF(lightVertices[li], cameraVertices[cip]);
//            cameraVertices[cip].reversePDF = evaluatePDF(cameraVertices[ci], lightVertices[li], cameraVertices[cip]);
//            DEBUG("cip. %f", (cameraVertices[cip].reversePDF));

//            if (cameraVertices[cip].reversePDF < 0)
//                DEBUG("cip li >= 0 negative. %f", cameraVertices[cip].reversePDF);

        } else { // l = 0 case
            constant Light& light = lights[cameraVertices[ci].si.lightIndex];
//            DEBUG("is light %d, light idx %d, type %d", cameraVertices[ci].isLight(), cameraVertices[ci].si.lightIndex, cameraVertices[ci].type);
            cameraVertices[cip].reversePDF = cameraVertices[ci].lightDirectionPDF(light, cameraVertices[cip]);
//            cameraVertices[cip].reversePDF = evaluateLightPDF(cameraVertices[ci], cameraVertices[cip]);
//            DEBUG("cip else. %f", cameraVertices[cip].reversePDF);

//            if (cameraVertices[cip].reversePDF < 0)
//                DEBUG("cip else negative. %f", cameraVertices[cip].reversePDF);

        }
    }
    
    if (li >= 0) {
        lightVertices[li].reversePDF = cameraVertices[ci].PDF(cameraVertices[cip], lightVertices[li]);
//        lightVertices[li].reversePDF = evaluatePDF(cameraVertices[ci], cameraVertices[cip], lightVertices[li]);
        lightVertices[li].delta = false;
//        DEBUG("li . %f", lightVertices[li].reversePDF);

//        if (lightVertices[li].reversePDF < 0)
//            DEBUG("li . %f", lightVertices[li].reversePDF);

    }
    
    if (lip >= 0) {
        lightVertices[lip].reversePDF = lightVertices[li].PDF(cameraVertices[ci], lightVertices[lip]);
//        lightVertices[lip].reversePDF = evaluatePDF(lightVertices[li], cameraVertices[ci], lightVertices[lip]);
//        DEBUG("lip. %f", lightVertices[lip].reversePDF);

//        if (lightVertices[li].reversePDF < 0)
//            DEBUG("lip negative. %f", lightVertices[lip].reversePDF);
    }
    
    float sum = 0.0f;
    float r = 1.0f;

    for (int i = ci; i > 0; i--) { // IF NO C = 1, i > 1 SINCE WE DONT COUNT C = 1 strategy so we dont include it in MIS
        r *= remap0(cameraVertices[i].reversePDF) / remap0(cameraVertices[i].forwardPDF);
//        DEBUG("camera - r: %f, rev: %f, fwd: %f, ratio: %f", r, remap0(cameraVertices[i].reversePDF), remap0(cameraVertices[i].forwardPDF), remap0(cameraVertices[i].reversePDF) / remap0(cameraVertices[i].forwardPDF));
//        if (i == 1) debug(cameraVertices[i].delta);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sum += r;
    }

    r = 1.0f;
    
    for (int i = li; i >= 0; i--) {
        
        r *= remap0(lightVertices[i].reversePDF) / remap0(lightVertices[i].forwardPDF);
        bool prevDelta = (i > 0) ? lightVertices[i - 1].delta : lightVertices[0].ei.light->delta;

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
//    if (c == 1) {
//        cameraVertices[0] = origVx;
//    }
    
//    debug(sum);

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
    
    if (uv.x < 0.0f || uv.x > 1.0f ||
        uv.y < 0.0f || uv.y > 1.0f)
    {
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
                       thread HaltonSampler& haltonSampler,
                       thread PathVertex *cameraVertices,
                       thread PathVertex *lightVertices,
                       int c, int l
                       )
{
    
//    if (lightVertices[l - 1].type == CAMERA_VERTEX) {
//        DEBUG("eded, %d", l);
//    }
//    if (lightVertices[0].type == LIGHT_VERTEX) {
//        DEBUG("!!!!");
//    }
//
    float3 contribution = float3(0.0f);
    PathVertex sampled;
    
    if (l == 0) { // l == 0 , l == 1 cases wrong in MIS likely light issue.
//        return contribution;
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        if (cameraVertex.isLight()) {
            contribution = lights[cameraVertex.si.lightIndex].color * cameraVertex.throughput;
        } else {
            return contribution;
        }
    } else if (c == 1) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        thread PathVertex& lightVertex = lightVertices[l - 1];
        constant Camera& camera = uniforms.camera;
//        return contribution;
        if (lightVertex.isConnectible()) {
            float3 condir = normalize(lightVertex.position() - cameraVertex.position());
            float d2 = length_squared(lightVertex.position() - cameraVertex.position());
//            sampled = createCameraVertex(camera, camera.position, camera.forward, float3(1.0f));
//            sampled.forwardPDF = 1.0f;
//            sampled.reversePDF = 0.0f;
//            float pdf = 1 / dot(condir, uniforms.camera.forward);
//            if (dot(uniforms.camera.forward, condir) < 0.0f) return float3(0.0f);
            float cameracos = dot(condir, camera.forward);
            float lightcos = dot(-condir, lightVertex.normal());
//            float fixedcos = cos(CAMERA_FOV_ANGLE * 0.5 * M_PI_F / 180.0f);
            float3 We = 1 / (A * pow(cameracos, 4));
            
//            DEBUG("cos: %f, We: %f", dot(condir, uniforms.camera.forward), We.x);

            We *= abs(lightcos / (d2));
//            debug(cos(CAMERA_FOV_ANGLE * 0.5 * M_PI_F / 180.0f));
//            if (cameracos < 0.0f || lightcos < 0.0f) {
//                return contribution;
//            }
//
            
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex) * lightVertex.si.textureColor;
            contribution = We * lightVertex.throughput * lightBSDF;
            
            contribution *= isVisible(cameraVertex.position(), cameraVertex.forwardPDF, lightVertex.position(), lightVertex.normal(), resources, instances, accelerationStructure);
            
//            if (!isBlack(contribution)) {
//                if (cameracos < 0.0f) {
//                    DEBUG("-cameracos %f", cameracos);
//                }
//                if (lightcos < 0.0f) {
//                    DEBUG("-lightcos %f", lightcos);
//                }
//            }

            
//            contribution *= isVisible(cameraVertex.position(), cameraVertex.forwardPDF, lightVertex.position(), lightVertex.normal(), resources, instances, accelerationStructure);

//            if (!isBlack(contribution))
//                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, resources, instances, accelerationStructure);
        }
    } else if (l == 1) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
//        return contribution;
        if (cameraVertex.isConnectible()) {
            float selectionPDF;
            constant Light& light = selectLight(lights, lightTriangles, uniforms, haltonSampler.r(), selectionPDF);
            LightSample lightSample = sampleLight(cameraVertex.position(), light, lightTriangles, environmentMapTexture, environmentMapCDF, haltonSampler.r3());
//            debug(cameraVertex.position(), lightSample.position);
            sampled = createLightVertex(light, lightSample.position, lightSample.normal, lightSample.emission / (selectionPDF * lightSample.PDF), selectionPDF * lightSample.PDF);
            
//            float3 condir = normalize(sampled.position() - cameraVertex.position());
//
//            if (dot(condir, cameraVertex.normal()) < 0.0f || dot(-condir, sampled.normal()) < 0.0f) { // FIXME: needs to be fixed for transmission
//                return float3(0.0f);
//            }
            
            contribution = 1 * cameraVertex.throughput * cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), sampled) * sampled.throughput * cameraVertex.si.textureColor;
            
            if (any(contribution < 0.0f)) {
                DEBUG("negative contributon");
            }
            
            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, sampled, resources, instances, accelerationStructure);
            
//            debug(contribution);
        }
    
    } else {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        thread PathVertex& lightVertex = lightVertices[l - 1];
//        return contribution;
        if (cameraVertex.isConnectible() && lightVertex.isConnectible()) {
            if (cameraVertex.type == CAMERA_VERTEX) DEBUG("camera: camera");
            if (cameraVertex.type == LIGHT_VERTEX) DEBUG("camera: light");
            //        if (cameraVertex.type == SURFACE_VERTEX) DEBUG("camera: surface");
            if (lightVertex.type == LIGHT_VERTEX) DEBUG("light: light, %d", l);
            //        if (lightVertex.type == SURFACE_VERTEX) DEBUG("light: surface, %d", l);
            if (lightVertex.type == CAMERA_VERTEX) DEBUG("light: camera, %d", l);
            
//            float3 condir = normalize(lightVertex.position() - cameraVertex.position());
//            if (dot(condir, cameraVertex.normal()) < 0.0f || dot(-condir, lightVertex.normal()) < 0.0f) { // FIXME: needs to be fixed for transmission
//                return float3(0.0f);
//            }
            
            float3 cameraBSDF = cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), lightVertex) * cameraVertex.si.textureColor;
            //        getBXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), connectionDirection, cameraVertex.normal(),                                      cameraVertex.getSurfaceInteraction().material) * cameraVertex.getSurfaceInteraction().textureColor;
            
            //        debug(cameraVertex.getSurfaceInteraction().material.BXDFs);
            //        debug(dot(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), connectionDirection));
            //        if (cameraVertex.getSurfaceInteraction().hitLight) {
            //            DEBUG("hti ligt");
            //        }
            
            //        if (length(cameraVertex.getSurfaceInteraction().textureColor) < 0.1)
            //            debug(cameraVertex.getSurfaceInteraction().textureColor);
            
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex) * lightVertex.si.textureColor;
            //        getBXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), -connectionDirection, lightVertex.normal(),                                      lightVertex.getSurfaceInteraction().material) * lightVertex.getSurfaceInteraction().textureColor; // fix for point lights
            //        debug(cameraBSDF);
            contribution = cameraVertex.throughput * lightVertex.throughput * cameraBSDF * lightBSDF;
//            debug(contribution);
//            DEBUG("c: %d, l: %d, mag: %f", c, l, length(contribution));
            if (any(contribution < 0.0f)) {
                DEBUG("negative contributon");
            }
            
            if (any(contribution > 1e-10f))
                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, resources, instances, accelerationStructure);
        }
    }
    
    float MISWeight = !isBlack(contribution) ? calculateMISWeight(uniforms, lights, cameraVertices, lightVertices, c, l, sampled) : 0.0f;
    contribution *= (MISWeight);
    
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
                                   thread HaltonSampler& haltonSampler,
                                   texture2d<float, access::read_write> splatTex,
                                   device atomic_float* splatBuffer
                                   )
{
    PathVertex cameraVertices[MAX_CAMERA_PATH_LENGTH];
    PathVertex lightVertices[MAX_LIGHT_PATH_LENGTH];
    
    int cameraPathLength = traceCameraPath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, haltonSampler, cameraVertices);
    
    int lightPathLength = traceLightPath(pixel, uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, haltonSampler, lightVertices);
//    debug(cameraVertices[1].getSurfaceInteraction().material.BXDFs);

//    if (lightPathLength >= 2 && lightVertices[1])
//    DEBUG("cam path length: %d, light path length: %d", cameraPathLength, lightPathLength);
    float3 totalContribution = float3(0.0f);
    
    for (int c = 1; c <= cameraPathLength; c++) {
        for (int l = 0; l <= lightPathLength; l++) {
            int depth = c + l - 2;
            if ((c == 1 && l == 1) || depth < 0 || depth > MAX_PATH_LENGTH)
                continue;
            
//            if (l == 1) continue;
                    
            float3 contribution = connectVertices(uniforms, resourceStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, haltonSampler, cameraVertices, lightVertices, c, l);
            
//            if (any(isnan(contribution)) || any(isinf(contribution))) {
//                DEBUG("c: %d, l: %d nan/inf", c, l);
//                continue;
//            }
            
            

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
