//
//  Lights.h
//  PathTracing
//
//  Created on 7/19/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Interactions.h"

#define MAX_LIGHTS 16
#define SCENE_RADIUS 8.0f

#define ENVIRONMENT_MAP_HEIGHT 2048
#define ENVIRONMENT_MAP_WIDTH 4096
#define ENVIRONMENT_MAP_SCALE 2

#define DELTA_LIGHT (1 << 7)

enum LightType : uint8_t {
    POINT_LIGHT = 0 | DELTA_LIGHT,
    AREA_LIGHT = 1,
    DIRECTIONAL_LIGHT = 2 | DELTA_LIGHT,
    ENVIRONMENT_MAP = 3
};

struct PointLight {
    vector_float3 position;
    vector_float3 color;
};

struct AreaLight {
    unsigned int firstTriangleIndex;
    unsigned int triangleCount;
    float totalArea;
};

struct DirectionalLight {
    vector_float3 direction;
    vector_float3 color;
};

struct EnvironmentMap {
    unsigned int width;
    unsigned int height;
};

struct Light {
    enum LightType type;
    unsigned int index;
    
    union {
        struct PointLight point;
        struct AreaLight area;
        struct DirectionalLight directional;
        struct EnvironmentMap environment;
    };
    
#ifdef __METAL_VERSION__
    bool isDelta() constant {
        return type & DELTA_LIGHT;
    }
#endif
};

struct LightTriangle {
    vector_float3 v0, v1, v2;
    vector_float2 uv0, uv1, uv2;
    vector_float3 emission;
    int emissionTextureIndex;
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

struct LightSample {
    float3 position;
    float3 normal;
    float3 wo;
    float3 emission;
    float distance;
    float PDF;
    
    LightSample(float3 _position, float3 _normal, float3 _wo, float3 _emission, float _distance, float _PDF) {
        position = _position;
        normal = _normal;
        wo = _wo;
        emission = _emission;
        distance = _distance;
        PDF = _PDF;
    }
};

struct LightEmissionSample {
    float3 position;
    float3 normal;
    float3 wo;
    float3 emission;
    float positionPDF;
    float directionPDF;
    
    LightEmissionSample(float3 _position, float3 _normal, float3 _wo, float3 _emission, float _positionPDF, float _directionPDF) {
        position = _position;
        normal = _normal;
        wo = _wo;
        emission = _emission;
        positionPDF = _positionPDF;
        directionPDF = _directionPDF;
    }
};

float environmentLightSamplePDF(constant Light& environmentMap, constant AliasEntry* environmentMapAliasEntries, float2 uv);

float getLightSelectionPDF(constant Light& light, constant AliasEntry* lightAliasEntries);

float getLightSamplePDF(constant Light& light);

LightSample sampleLight(float3 position,
                        constant Light& light,
                        constant LightTriangle *lightTriangles,
                        constant Textures* textures,
                        texture2d<float> environmentMapTexture,
                        constant AliasEntry* lightTriangleAliasEntries,
                        constant AliasEntry* environmentMapAliasEntries,
                        thread Sampler& sampler);

constant Light& selectLight(constant Light* lights,
                            constant AliasEntry* lightAliasEntries,
                            constant Uniforms& uniforms,
                            thread float& selectionPDF,
                            float2 r2);

LightEmissionSample sampleLightEmission(constant Light& light,
                                        constant LightTriangle *lightTriangles,
                                        constant Textures* textures,
                                        texture2d<float> environmentMapTexture,
                                        constant AliasEntry* lightTriangleAliasEntries,
                                        constant AliasEntry* environmentMapAliasEntries,
                                        thread Sampler& sampler);

float getLightDirectionPDF(float3 w, float3 n, constant Light& light, constant AliasEntry* environmentMapAliasEntries);

float3 environmentMapEmission(texture2d<float> environmentMapTexture, float2 uv);

inline float2 getEnvironmentMapUV(float3 w) {
    float u = (atan2(w.z, w.x) + M_PI_F) / (2.0f * M_PI_F);
    float v = 1.0f - (asin(clamp(w.y, -1.0f, 1.0f)) + M_PI_2_F) / M_PI_F;
    return float2(u, v);
}

float infiniteLightDensity(float3 w, constant Light& environmentLight, constant AliasEntry* lightAliasEntries, constant AliasEntry* environmentMapAliasEntries);

#endif
