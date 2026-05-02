//
//  MegaKernel.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "BDPT.h"

using namespace metal;
using namespace raytracing;

int tracePath(float2 pixel,
              constant Uniforms& uniforms,
              device MTLAccelerationStructureInstanceDescriptor *instances,
              instance_acceleration_structure accelerationStructure,
              constant Light *lights,
              constant LightTriangle *lightTriangles,
              constant int* instanceLightIndices,
              constant Textures* textures,
              thread Sampler& sampler,
              ray ray,
              int maxDepth,
              TransportMode transportMode,
              thread PathVertex *vertices,
              VertexType type,
              float3 throughput,
              float forwardPDF,
              constant Material* materials,
              constant AliasEntry* lightAliasEntries,
              constant AliasEntry* lightTriangleAliasEntries,
              texture2d<float> environmentMapTexture,
              constant AliasEntry* environmentMapAliasEntries)
{
    int bounces = 1;
    
    for (int bounce = 1; bounce < maxDepth; bounce++) {
        thread PathVertex& vx = vertices[bounces];
        thread PathVertex& prev = vertices[bounces - 1];
        
        IntersectionResult intersection = intersect<false>(ray, accelerationStructure);
        
        if (intersection.type == intersection_type::none) {
            if (uniforms.environmentMapLightIndex == -1)
                break;

            if (type == CAMERA_VERTEX) {
                float3 endPos = ray.origin + ray.direction * 4 * SCENE_RADIUS;
                vx = createLightVertex(nullptr, endPos, float3(0.0f), throughput, forwardPDF);
                bounces++;
            }
            
            break;
        }

        SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, instances, accelerationStructure, instanceLightIndices, textures, materials);
        SampledMaterial material = surfaceInteraction.material;
        
        if (material.BXDFs == DIFFUSE || material.BXDFs == CONDUCTOR || material.BXDFs == DIELECTRIC_REFLECTION) {
            if (dot(-ray.direction, surfaceInteraction.normal) < 0.0f)
                surfaceInteraction.normal = -surfaceInteraction.normal;
        }
                
        if (material.alphaMode == ALPHA_MASK && material.alpha < material.alphaCutoff) {
            ray.origin = surfaceInteraction.position + ray.direction * 1e-4f;
            continue;
        }

        float3 n = surfaceInteraction.normal;
        
        vx = createSurfaceVertex(surfaceInteraction, throughput, forwardPDF, prev);

        if (++bounces >= maxDepth) {
            break;
        }
                
        BSDFSample bsdfSample = sampleBXDF(-ray.direction, n, material, transportMode, sampler.r3());
        
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

        ray.origin = surfaceInteraction.position + calculateOffset(wo, n, 1e-4f);
        ray.direction = wo;
        ray.min_distance = 1e-5f;
    }

    return bounces;
}

int traceCameraPath(float2 pixel,
                    constant Uniforms& uniforms,
                    device MTLAccelerationStructureInstanceDescriptor *instances,
                    instance_acceleration_structure accelerationStructure,
                    constant Light *lights,
                    constant LightTriangle *lightTriangles,
                    constant int* instanceLightIndices,
                    constant Textures* textures,
                    thread Sampler& sampler,
                    thread PathVertex *cameraVertices,
                    constant Material* materials,
                    constant AliasEntry* lightAliasEntries,
                    constant AliasEntry* lightTriangleAliasEntries,
                    texture2d<float> environmentMapTexture,
                    constant AliasEntry* environmentMapAliasEntries)
{
    constant Camera& camera = uniforms.camera;
    ray ray = generateRay(pixel, uniforms);
    
    float positionPDF, directionPDF;
    cameraRayPDF(uniforms, ray.direction, positionPDF, directionPDF);
    
    cameraVertices[0] = createCameraVertex(&camera, camera.position, camera.forward, float3(1.0f));
    cameraVertices[0].forwardPDF = positionPDF;
    cameraVertices[0].delta = true;

    float3 throughput = float3(1.0f);

    return tracePath(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, sampler, ray, MAX_CAMERA_PATH_LENGTH, Radiance, cameraVertices, CAMERA_VERTEX, throughput, directionPDF, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);
}

int traceLightPath(float2 pixel,
                   constant Uniforms& uniforms,
                   device MTLAccelerationStructureInstanceDescriptor *instances,
                   instance_acceleration_structure accelerationStructure,
                   constant Light *lights,
                   constant LightTriangle *lightTriangles,
                   constant int* instanceLightIndices,
                   constant Textures* textures,
                   thread Sampler& sampler,
                   thread PathVertex *lightVertices,
                   constant Material* materials,
                   constant AliasEntry* lightAliasEntries,
                   constant AliasEntry* lightTriangleAliasEntries,
                   texture2d<float> environmentMapTexture,
                   constant AliasEntry* environmentMapAliasEntries)
{
    float selectionPDF;
    constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
    LightEmissionSample lightEmissionSample = sampleLightEmission(light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);
    float positionPDF = lightEmissionSample.positionPDF;
    float directionPDF = lightEmissionSample.directionPDF;
    float3 normal = lightEmissionSample.normal;

    ray ray;
    float epsilon = calculateEpsilon(lightEmissionSample.position);
    ray.origin = lightEmissionSample.position + calculateOffset(lightEmissionSample.wo, normal, epsilon);
    ray.direction = lightEmissionSample.wo;
    ray.min_distance = epsilon;
    ray.max_distance = INFINITY;
    
    lightVertices[0] = createLightVertex(&light, lightEmissionSample.position, normal, lightEmissionSample.emission, positionPDF * selectionPDF);
    float3 throughput = lightEmissionSample.emission / (selectionPDF * positionPDF * directionPDF);

    if (light.type == AREA_LIGHT)
        throughput *= abs(dot(lightVertices[0].normal(), ray.direction));

    int numVertices = tracePath(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, sampler, ray, MAX_LIGHT_PATH_LENGTH, Importance, lightVertices, LIGHT_VERTEX, throughput, directionPDF, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);

    if (lightVertices[0].isInfiniteLight()) {
        if (numVertices > 0) {
            lightVertices[1].forwardPDF = positionPDF;
            if (lightVertices[1].isOnSurface())
                lightVertices[1].forwardPDF *= abs(dot(ray.direction, lightVertices[1].normal()));
        }
        lightVertices[0].forwardPDF = infiniteLightDensity(-ray.direction, lights[uniforms.environmentMapLightIndex], lightAliasEntries, environmentMapAliasEntries);
    }
    
    return numVertices;
}

float3 calculateGeometricTerm(thread PathVertex& cameraVertex,
                              thread PathVertex& lightVertex,
                              device MTLAccelerationStructureInstanceDescriptor *instances,
                              instance_acceleration_structure accelerationStructure
                              )
{
    float3 connectionVector = lightVertex.position() - cameraVertex.position();
    float connectionDistance = length(connectionVector);
    float3 connectionDirection = connectionVector / connectionDistance;
                
    bool noFalloff = lightVertex.isInfiniteLight()
                 || (lightVertex.type == LIGHT_VERTEX && lightVertex.ei.light && lightVertex.ei.light->type == DIRECTIONAL_LIGHT);

    float G = noFalloff ? 1.0f : 1.0f / max(connectionDistance * connectionDistance, 1e-8f);
    
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
    
    if (!isVisible(cameraVertex.position(), cameraVertex.normal(), lightVertex.position(), lightVertex.normal(), instances, accelerationStructure)) {
        return float3(0.0f);
    }

    return G;
}

float calculateMISWeight(constant Uniforms& uniforms,
                         constant Light* lights,
                         constant AliasEntry* lightAliasEntries,
                         constant AliasEntry* lightTriangleAliasEntries,
                         texture2d<float> environmentMapTexture,
                         constant AliasEntry* environmentMapAliasEntries,
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
    bool origCamDelta = cameraVertices[ci].delta;
    float origCamPrevRev = (cip >= 0) ? cameraVertices[cip].reversePDF : 0.0f;
    bool origCamPrevDel = (cip >= 0) ? cameraVertices[cip].delta   : false;

    float origLgtRev = (li >= 0) ? lightVertices[li].reversePDF : 0.0f;
    bool origLgtDelta = (li >= 0) ? lightVertices[li].delta : false;
    float origLgtPrevRev = (lip >= 0) ? lightVertices[lip].reversePDF : 0.0f;
    bool origLgtPrevDel = (lip >= 0) ? lightVertices[lip].delta   : false;
                
    if (ci >= 0) {
        if (li >= 0) {
            cameraVertices[ci].reversePDF = lightVertices[li].PDF(lightVertices[lip], cameraVertices[ci], environmentMapAliasEntries, uniforms);
        } else { // l = 0 case
            constant Light& light = lights[cameraVertices[ci].si.lightIndex]; // originally origin
            cameraVertices[ci].reversePDF = cameraVertices[ci].lightOriginPDF(light, lights, uniforms, cameraVertices[cip], lightAliasEntries, environmentMapAliasEntries);
        }
        cameraVertices[ci].delta = false;
    }
    
    if (cip >= 0) {
        if (li >= 0) {
            cameraVertices[cip].reversePDF = cameraVertices[ci].PDF(lightVertices[li], cameraVertices[cip], environmentMapAliasEntries, uniforms);

        } else { // l = 0 case
            constant Light& light = lights[cameraVertices[ci].si.lightIndex]; // originally direction
            cameraVertices[cip].reversePDF = cameraVertices[ci].lightDirectionPDF(light, cameraVertices[cip], environmentMapAliasEntries);

        }
    }
    
    if (li >= 0) {
        lightVertices[li].reversePDF = cameraVertices[ci].PDF(cameraVertices[cip], lightVertices[li], environmentMapAliasEntries, uniforms);
        lightVertices[li].delta = false;
    }
    
    if (lip >= 0) {
        lightVertices[lip].reversePDF = lightVertices[li].PDF(cameraVertices[ci], lightVertices[lip], environmentMapAliasEntries, uniforms);
    }
    
    float sum = 0.0f;
    float r = 1.0f;

    for (int i = ci; i > 0; i--) {
        r *= remap0(cameraVertices[i].reversePDF) / remap0(cameraVertices[i].forwardPDF);

        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sum += r * r;
    }

    r = 1.0f;
    
    for (int i = li; i >= 0; i--) {
        r *= remap0(lightVertices[i].reversePDF) / remap0(lightVertices[i].forwardPDF);
        bool prevDelta = (i > 0) ? lightVertices[i - 1].delta : lightVertices[0].isDeltaLight();

        if (!lightVertices[i].delta && !prevDelta)
            sum += r * r;
    }
        
    cameraVertices[ci].reversePDF = originalCameraReverse;
    cameraVertices[ci].delta = origCamDelta;
    if (cip >= 0) {
        cameraVertices[cip].reversePDF = origCamPrevRev;
        cameraVertices[cip].delta = origCamPrevDel;
    }

    if (li >= 0) {
        lightVertices[li].reversePDF = origLgtRev;
        lightVertices[li].delta = origLgtDelta;
    }
    if (lip >= 0) {
        lightVertices[lip].reversePDF = origLgtPrevRev;
        lightVertices[lip].delta = origLgtPrevDel;
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

void splat(constant Uniforms& uniforms,
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
                       device MTLAccelerationStructureInstanceDescriptor *instances,
                       instance_acceleration_structure accelerationStructure,
                       constant Light *lights,
                       constant LightTriangle *lightTriangles,
                       constant int *lightIndices,
                       constant Textures *textures,
                       constant AliasEntry* lightAliasEntries,
                       constant AliasEntry* lightTriangleAliasEntries,
                       texture2d<float> environmentMapTexture,
                       constant AliasEntry* environmentMapAliasEntries,
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
            float3 We = cameraWe(uniforms, lightVertex.position());
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex, Importance);
            
            contribution = We * lightVertex.throughput * lightBSDF;
            
            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, instances, accelerationStructure);
        }
    } else if (l == 1) {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
//        return contribution;
        if (cameraVertex.isConnectible()) {
            float selectionPDF;
            constant Light& light = selectLight(lights, lightAliasEntries, uniforms, selectionPDF, sampler.r2());
            LightSample lightSample = sampleLight(cameraVertex.position(), light, lightTriangles, textures, environmentMapTexture, lightTriangleAliasEntries, environmentMapAliasEntries, sampler);
            sampled = createLightVertex(&light, lightSample.position, lightSample.normal, lightSample.emission / (selectionPDF * lightSample.PDF), selectionPDF * lightSample.PDF);

            contribution = cameraVertex.throughput * cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), sampled, Radiance) * sampled.throughput;
                        
            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, sampled, instances, accelerationStructure);
        }
    } else {
        thread PathVertex& cameraVertex = cameraVertices[c - 1];
        thread PathVertex& lightVertex = lightVertices[l - 1];
//        return contribution;
        if (cameraVertex.isConnectible() && lightVertex.isConnectible()) {
            float3 cameraBSDF = cameraVertex.BXDF(-normalize(cameraVertex.position() - cameraVertices[c - 2].position()), lightVertex, Radiance);
            float3 lightBSDF = lightVertex.BXDF(-normalize(lightVertex.position() - lightVertices[l - 2].position()), cameraVertex, Importance);
            
            contribution = cameraVertex.throughput * lightVertex.throughput * cameraBSDF * lightBSDF;

            if (!isBlack(contribution))
                contribution *= calculateGeometricTerm(cameraVertex, lightVertex, instances, accelerationStructure);
        }
    }
    
    float MISWeight = !isBlack(contribution) ? calculateMISWeight(uniforms, lights, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries, cameraVertices, lightVertices, c, l, sampled) : 0.0f;
    contribution *= MISWeight;

    return contribution;
}

float3 bidirectionalPathIntegrator(float2 pixel,
                                   constant Uniforms& uniforms,
                                   device MTLAccelerationStructureInstanceDescriptor *instances,
                                   instance_acceleration_structure accelerationStructure,
                                   constant Light *lights,
                                   constant LightTriangle *lightTriangles,
                                   constant int* instanceLightIndices,
                                   constant Textures* textures,
                                   thread Sampler& sampler,
                                   device atomic_float* splatBuffer,
                                   constant Material* materials,
                                   constant AliasEntry* lightAliasEntries,
                                   constant AliasEntry* lightTriangleAliasEntries,
                                   texture2d<float> environmentMapTexture,
                                   constant AliasEntry* environmentMapAliasEntries)
{
    PathVertex cameraVertices[MAX_CAMERA_PATH_LENGTH];
    PathVertex lightVertices[MAX_LIGHT_PATH_LENGTH];
    
    int cameraPathLength = traceCameraPath(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, sampler, cameraVertices, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);
    
    int lightPathLength = traceLightPath(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, sampler, lightVertices, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);

    float3 totalContribution = float3(0.0f);
    
    for (int c = 1; c <= cameraPathLength; c++) {
        for (int l = 0; l <= lightPathLength; l++) {
            int depth = c + l - 2;
            if ((c == 1 && l == 1) || depth < 0 || depth > MAX_PATH_LENGTH)
                continue;

            float3 contribution = connectVertices(uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries, sampler, cameraVertices, lightVertices, c, l);
            
            if (any(contribution < 0.0f) || any(isnan(contribution)) || any(isinf(contribution))) {
//                DEBUG("Invalid contribution - c: %d, l: %d, float3(%f, %f, %f)", c, l, contribution.x, contribution.y, contribution.z);
                continue;
            }

            if (c == 1) {
                uint2 pixelCoord = projectToScreen(lightVertices[l - 1].position(), uniforms);
                splat(uniforms, pixelCoord, contribution, splatBuffer);
            } else {
                totalContribution += contribution;
            }
        }
    }
    
    return totalContribution;
}

kernel void bidirectionalPathTracingKernel(device float* accumulation,
                                           device atomic_float* splatBuffer,
                                           
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
                                           
                                           constant uint* sobolValues,
                             
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
    
    float3 contribution = bidirectionalPathIntegrator(pixel, uniforms, instances, accelerationStructure, lights, lightTriangles, instanceLightIndices, textures, sampler, splatBuffer, materials, lightAliasEntries, lightTriangleAliasEntries, environmentMapTexture, environmentMapAliasEntries);
    
    addContribution(contribution, tid.y * uniforms.width + tid.x, accumulation);
}

kernel void finalizeAccumulationBDPT(device float* accumulation,
                                     device float* splatAccmulation,
                                     constant Uniforms& uniforms,
                                     texture2d<float, access::read_write> finalImage,
                                     
                                     uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    uint pixelIndex = tid.y * uniforms.width + tid.x;
    float3 contribution = getFloat3FromAccumulation(pixelIndex, accumulation);
    float3 splatContribution = getFloat3FromAccumulation(pixelIndex, splatAccmulation);
    
    float3 color = (contribution + splatContribution) / (uniforms.frameIndex + 1u);
    finalImage.write(float4(color, 1.0f), tid);
}
