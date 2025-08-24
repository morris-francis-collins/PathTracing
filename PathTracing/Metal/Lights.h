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
#define SCENE_RADIUS 4.5f

#define ENVIRONMENT_MAP_HEIGHT 2048
#define ENVIRONMENT_MAP_WIDTH 4096
#define ENVIRONMENT_MAP_SCALE 2

enum LightType : unsigned int {
    POINT_LIGHT = 0,
    AREA_LIGHT = 1,
    DIRECTIONAL_LIGHT = 2,
    ENVIRONMENT_MAP = 3
};

struct Light {
    enum LightType type;
    unsigned int index;
    int delta;
    vector_float3 position;
    vector_float3 color;
    unsigned int firstTriangleIndex; // area lights only
    unsigned int triangleCount;
    float totalArea;
    vector_float3 direction; // directional lights only
};

struct LightTriangle {
    vector_float3 v0;
    vector_float3 v1;
    vector_float3 v2;
    vector_float3 emission0;
    vector_float3 emission1;
    vector_float3 emission2;
    float area;
    float cdf;
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

float environmentLightSamplePDF(float2 uv, constant float *environmentMapCDF);

float getLightSelectionPDF(constant Light& light, constant Light *lights, constant Uniforms& uniforms);

float getLightSamplePDF(constant Light& light);

LightSample sampleAreaLight(constant Light& areaLight, constant LightTriangle *lightTriangles, float3 r3);

LightSample sampleLight(float3 position,
                        constant Light& light,
                        constant LightTriangle *lightTriangles,
                        texture2d<float> environmentMapTexture,
                        constant float *environmentMapCDF,
                        float3 r3
                        );

constant Light& selectLight(constant Light *lights, constant LightTriangle *lightTriangles, constant Uniforms& uniforms, float r, thread float& selectionPDF);

LightEmissionSample sampleLightEmission(constant Light& light,
                                        constant LightTriangle *lightTriangles,
                                        texture2d<float> environmentMapTexture,
                                        constant float *environmentMapCDF,
                                        float2 r2,
                                        float3 r3
                                        );

float getLightDirectionPDF(constant Light& light, float3 w, float3 n, constant float* environmentMapCDF);

float3 environmentMapEmission(float2 uv, texture2d<float> environmentMapTexture);

inline float2 getEnvironmentMapUV(float3 w) {
    float u = (atan2(w.z, w.x) + M_PI_F) / (2.0f * M_PI_F);
    float v = 1.0f - (asin(clamp(w.y, -1.0f, 1.0f)) + M_PI_2_F) / M_PI_F;
    return float2(u, v);
}

float infiniteLightDensity(float3 w, constant Light* lights, constant Uniforms& uniforms, constant float *environmentMapCDF);

#endif
