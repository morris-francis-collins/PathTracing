//
//  Lights.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include "Lights.h"
#include "Interactions.h"

using namespace metal;
using namespace raytracing;

// MARK: Point Lights

LightSample samplePointLight(float3 position, constant Light& pointLight) {
    float3 wo = pointLight.point.position - position;
    float d = length(wo);
    wo /= d;
    
    return LightSample(pointLight.point.position, float3(0.0f), wo, pointLight.point.color, d, 1.0f);
}

LightEmissionSample samplePointLightEmission(constant Light& pointLight, float2 r2) {
    float z = 1.0f - 2.0f * r2.x;
    float phi = 2.0f * M_PI_F * r2.y;
    float radius = sqrt(max(1.0f - z * z, 0.0f));
    float3 wo = float3(radius * cos(phi), radius * sin(phi), z);
    float directionPDF = 1.0f / (4.0f * M_PI_F);
    
    return LightEmissionSample(pointLight.point.position, float3(0.0f), wo, pointLight.point.color, 1.0f, directionPDF);
}

// MARK: Area Lights

static LightTriangle getLightTriangle(constant Light& areaLight, constant LightTriangle *lightTriangles, constant AliasEntry* lightTriangleAliasEntries, float2 r2) {
    unsigned int index = areaLight.area.firstTriangleIndex + clamp(uint(areaLight.area.triangleCount * r2.x), 0u, areaLight.area.triangleCount - 1u);
    
    if (lightTriangleAliasEntries[index].acceptanceProbability > r2.y) {
        return lightTriangles[index];
    } else {
        return lightTriangles[lightTriangleAliasEntries[index].alias];
    }
}

static void getTriangleSamplingCoordinates(float2 r2, thread float& u, thread float& v, thread float& w) {
    if (r2.x + r2.y > 1.0f) {
        r2.x = 1.0f - r2.x;
        r2.y = 1.0f - r2.y;
    }

    u = 1.0f - r2.x - r2.y;
    v = r2.x;
    w = r2.y;
}

static float3 sampleEmissionTexture(constant Textures *textures, thread LightTriangle& triangle, float2 uv) {
    constexpr sampler textureSampler(address::repeat, filter::linear);
    return triangle.emissionTextureIndex != -1 ? textures->textures[triangle.emissionTextureIndex].sample(textureSampler, uv).rgb : float3(1.0f);
}

LightSample sampleAreaLight(float3 position,
                            constant Light& areaLight,
                            constant LightTriangle *lightTriangles,
                            constant AliasEntry* lightTriangleAliasEntries,
                            constant Textures* textures,
                            float2 r2_0,
                            float2 r2_1)
{
    LightTriangle triangle = getLightTriangle(areaLight, lightTriangles, lightTriangleAliasEntries, r2_0);
    
    float u, v, w;
    getTriangleSamplingCoordinates(r2_1, u, v, w);
    
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 lightPosition = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    float2 lightUV = u * triangle.uv0 + v * triangle.uv1 + w * triangle.uv2;
    float3 normal = normalize(cross(edge1, edge2));
        
    float3 wo = lightPosition - position;
    float d = length(wo);
    wo /= d;
    
    float3 emission = triangle.emission * sampleEmissionTexture(textures, triangle, lightUV);
    
    return LightSample(lightPosition, normal, wo, emission, d, 1.0f / areaLight.area.totalArea);
}

LightEmissionSample sampleAreaLightEmission(constant Light& areaLight,
                                            constant LightTriangle *lightTriangles,
                                            constant AliasEntry* lightTriangleAliasEntries,
                                            constant Textures* textures,
                                            float2 r2_0,
                                            float2 r2_1,
                                            float2 r2_2)
{
    LightTriangle triangle = getLightTriangle(areaLight, lightTriangles, lightTriangleAliasEntries, r2_0);
    
    float u, v, w;
    getTriangleSamplingCoordinates(r2_1, u, v, w);

    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 lightPosition = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    float2 lightUV = u * triangle.uv0 + v * triangle.uv1 + w * triangle.uv2;
    float3 normal = normalize(cross(edge1, edge2));

    float3 woLocal = sampleCosineWeightedHemisphere(r2_2);
    float3 wo = alignHemisphereWithNormal(woLocal, normal);
    
    float positionPDF = 1.0f / areaLight.area.totalArea;
    float directionPDF = dot(wo, normal) / M_PI_F;
    
    float3 emission = triangle.emission * sampleEmissionTexture(textures, triangle, lightUV);
    
    return LightEmissionSample(lightPosition, normal, wo, emission, positionPDF, directionPDF);
}

// MARK: Directional Lights

LightSample sampleDirectionalLight(float3 position, constant Light& directionalLight) {
    float3 wo = -directionalLight.directional.direction;
    float3 lightPosition = position + 2.0f * SCENE_RADIUS * wo;
    return LightSample(lightPosition, float3(0.0f), wo, directionalLight.directional.color, 1.0f, 1.0f);
}

LightEmissionSample sampleDirectionalLightEmission(constant Light& directionalLight, float2 r2) {
    float3 T, B; createOrthonormalBasis(directionalLight.directional.direction, T, B);
    float2 diskSample = concentricSampleDisk(r2) * SCENE_RADIUS;
    float3 position = float3(0.0f) - 2.0f * directionalLight.directional.direction * SCENE_RADIUS + T * diskSample.x + B * diskSample.y;
    float positionPDF = 1.0f / (M_PI_F * SCENE_RADIUS * SCENE_RADIUS);
    
    return LightEmissionSample(position, float3(0.0f), directionalLight.directional.direction, directionalLight.directional.color, positionPDF, 1.0f);
}

// MARK: Environment Maps

float3 environmentMapEmission(texture2d<float> environmentMapTexture, float2 uv) {
    constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);
    return environmentMapTexture.sample(textureSampler, uv).rgb;
}

struct EnvironmentMapSample {
    unsigned int index;
    float PMF;
};

static EnvironmentMapSample sampleEnvironmentMapAliasTable(constant Light& environmentMap, constant AliasEntry* environmentMapAliasEntries, float2 r2) {
    unsigned int pixels = environmentMap.environment.width * environmentMap.environment.height;
    unsigned int index = floor(pixels * r2.x);
    
    if (environmentMapAliasEntries[index].acceptanceProbability > r2.y) {
        return {index, environmentMapAliasEntries[index].PMF};
    } else {
        unsigned int aliasIndex = environmentMapAliasEntries[index].alias;
        return {aliasIndex, environmentMapAliasEntries[aliasIndex].PMF};
    }
}

float environmentLightSamplePDF(constant Light& environmentMap, constant AliasEntry* environmentMapAliasEntries, float2 uv) {
    unsigned int x = clamp(uint(uv.x * environmentMap.environment.width), 0u, environmentMap.environment.width - 1u);
    unsigned int y = clamp(uint(uv.y * environmentMap.environment.height), 0u, environmentMap.environment.height - 1u);
    unsigned int i = y * environmentMap.environment.width + x;
    
    float pixelPDF = environmentMapAliasEntries[i].PMF;
    return max(pixelPDF * float(environmentMap.environment.width * environmentMap.environment.height) / (2.0f * M_PI_F * M_PI_F * sin(uv.y * M_PI_F)), 1e-30f);
}

LightSample sampleEnvironmentMap(float3 position,
                                 constant Light& environmentMap,
                                 texture2d<float> environmentMapTexture,
                                 constant AliasEntry* environmentMapAliasEntries,
                                 float2 r2)
{
    EnvironmentMapSample environmentMapSample = sampleEnvironmentMapAliasTable(environmentMap, environmentMapAliasEntries, r2);
    
    int y = environmentMapSample.index / environmentMap.environment.width;
    int x = environmentMapSample.index % environmentMap.environment.width;

    float u = (float(x) + 0.5f) / float(environmentMap.environment.width);
    float v = (float(y) + 0.5f) / float(environmentMap.environment.height);
    
    float3 emission = environmentMapEmission(environmentMapTexture, float2(u, v));

    float phi = u * 2.0f * M_PI_F;
    float theta = v * M_PI_F;
    
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    
    float3 wo;
    wo.x = -sinTheta * cosPhi;
    wo.y = cosTheta;             // y is up
    wo.z = -sinTheta * sinPhi;
    
    float pixelPDF = environmentMapSample.PMF;
    float solidAnglePDF = pixelPDF * float(environmentMap.environment.width * environmentMap.environment.height) / (2.0f * M_PI_F * M_PI_F * sinTheta);
    float3 lightPosition = position + 2.0f * SCENE_RADIUS * wo;
    
    return LightSample(lightPosition, float3(0.0f), wo, emission, 1.0, solidAnglePDF); // TODO: 1.0 distance since we dont want fall off; adjust so infinte lights no fall off  for clarity
}

LightEmissionSample sampleEnvironmentMapEmission(constant Light& environmentMap,
                                                 texture2d<float> environmentMapTexture,
                                                 constant AliasEntry* environmentMapAliasEntries,
                                                 float2 r2_0,
                                                 float2 r2_1)
{
    EnvironmentMapSample environmentMapSample = sampleEnvironmentMapAliasTable(environmentMap, environmentMapAliasEntries, r2_0);
    
    int y = environmentMapSample.index / environmentMap.environment.width;
    int x = environmentMapSample.index % environmentMap.environment.width;

    float u = (float(x) + 0.5f) / float(environmentMap.environment.width);
    float v = (float(y) + 0.5f) / float(environmentMap.environment.height);

    float3 emission = environmentMapEmission(environmentMapTexture, float2(u, v));

    float phi = u * 2.0f * M_PI_F;
    float theta = v * M_PI_F;
    
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    float3 wo;
    wo.x = -sinTheta * cosPhi;
    wo.y = cosTheta;             // y is up
    wo.z = -sinTheta * sinPhi;
    
    float3 T, B; createOrthonormalBasis(wo, T, B);
    float2 diskSample = concentricSampleDisk(r2_1) * SCENE_RADIUS;
    float3 position = float3(0.0f) + 2.0f * wo * SCENE_RADIUS + T * diskSample.x + B * diskSample.y;

    float positionPDF = 1.0f / (M_PI_F * SCENE_RADIUS * SCENE_RADIUS);
    float pixelPDF = environmentMapSample.PMF;
    float directionPDF = pixelPDF * float(environmentMap.environment.width * environmentMap.environment.height) / (2.0f * M_PI_F * M_PI_F * sinTheta);
    
    return LightEmissionSample(position, float3(0.0f), -wo, emission, positionPDF, max(1e-30f, directionPDF));
}

// MARK: Other

constant Light& selectLight(constant Light* lights,
                            constant AliasEntry* lightAliasEntries,
                            constant Uniforms& uniforms,
                            thread float& selectionPDF,
                            float2 r2)
{
    unsigned int index = clamp(uint(uniforms.lightCount * r2.x), 0u, uniforms.lightCount - 1u);
    
    if (lightAliasEntries[index].acceptanceProbability > r2.y) {
        selectionPDF = lightAliasEntries[index].PMF;
        return lights[index];
    } else {
        unsigned int aliasIndex = lightAliasEntries[index].alias;
        selectionPDF = lightAliasEntries[aliasIndex].PMF;
        return lights[aliasIndex];
    }
}

float getLightSelectionPDF(constant Light& light, constant AliasEntry* lightAliasEntries) {
    return lightAliasEntries[light.index].PMF;
}

LightSample sampleLight(float3 position,
                        constant Light& light,
                        constant LightTriangle *lightTriangles,
                        constant Textures* textures,
                        texture2d<float> environmentMapTexture,
                        constant AliasEntry* lightTriangleAliasEntries,
                        constant AliasEntry* environmentMapAliasEntries,
                        thread Sampler& sampler)
{
    switch (light.type) {
        case POINT_LIGHT:
            return samplePointLight(position, light);
        case AREA_LIGHT:
            return sampleAreaLight(position, light, lightTriangles, lightTriangleAliasEntries, textures, sampler.r2(), sampler.r2());
        case DIRECTIONAL_LIGHT:
            return sampleDirectionalLight(position, light);
        case ENVIRONMENT_MAP:
            return sampleEnvironmentMap(position, light, environmentMapTexture, environmentMapAliasEntries, sampler.r2());
        default:
            return LightSample(float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f), 0.0f, 0.0f);
    }
}

float getLightSamplePDF(constant Light& light) {
    switch (light.type) {
        case POINT_LIGHT:
            return 0.0f;
        case AREA_LIGHT:
            return 1.0f / light.area.totalArea;
        case DIRECTIONAL_LIGHT:
            return 1.0f / (M_PI_F * SCENE_RADIUS * SCENE_RADIUS);
        case ENVIRONMENT_MAP:
            return 1.0f / (M_PI_F * SCENE_RADIUS * SCENE_RADIUS);
        default:
            return 0.0f;
    }
}

float getLightDirectionPDF(float3 w, float3 n, constant Light& light, constant AliasEntry* environmentMapAliasEntries) {
    switch (light.type) {
        case POINT_LIGHT:
            return 1.0f / (4.0f * M_PI_F);
        case AREA_LIGHT:
            return max(dot(w, n), 0.0f) / M_PI_F;
        case DIRECTIONAL_LIGHT:
            return 1.0f;
        case ENVIRONMENT_MAP:
            return environmentLightSamplePDF(light, environmentMapAliasEntries, getEnvironmentMapUV(w));
        default:
            DEBUG("getLightDirectionPDF: NOT IMPLEMENTED");
            return 0.0f;
    }
}

LightEmissionSample sampleLightEmission(constant Light& light,
                                        constant LightTriangle *lightTriangles,
                                        constant Textures* textures,
                                        texture2d<float> environmentMapTexture,
                                        constant AliasEntry* lightTriangleAliasEntries,
                                        constant AliasEntry* environmentMapAliasEntries,
                                        thread Sampler& sampler)
{
    switch (light.type) {
        case POINT_LIGHT:
            return samplePointLightEmission(light, sampler.r2());
        case AREA_LIGHT:
            return sampleAreaLightEmission(light, lightTriangles, lightTriangleAliasEntries, textures, sampler.r2(), sampler.r2(), sampler.r2());
        case DIRECTIONAL_LIGHT:
            return sampleDirectionalLightEmission(light, sampler.r2());
        case ENVIRONMENT_MAP:
            return sampleEnvironmentMapEmission(light, environmentMapTexture, environmentMapAliasEntries, sampler.r2(), sampler.r2());
        default:
            DEBUG("sampleLightEmission: NOT IMPLEMENTED");
            return LightEmissionSample(float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f), 0.0f, 0.0f);
    }
}

float infiniteLightDensity(float3 w, constant Light& environmentLight, constant AliasEntry* lightAliasEntries, constant AliasEntry* environmentMapAliasEntries) {
    float2 uv = getEnvironmentMapUV(w);
    return environmentLightSamplePDF(environmentLight, environmentMapAliasEntries, uv) * lightAliasEntries[environmentLight.index].PMF;
}
