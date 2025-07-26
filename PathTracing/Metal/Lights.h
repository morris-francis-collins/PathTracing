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
#define SCENE_RADIUS 8

#define ENVIRONMENT_MAP_HEIGHT 2048
#define ENVIRONMENT_MAP_WIDTH 4096
#define ENVIRONMENT_MAP_SCALE 5

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

float environmentLightSamplePDF(float2 uv, device float *environmentMapCDF);

float getLightSelectionPDF(device Light *lights, constant Uniforms& uniforms, unsigned int lightIndex);

float getLightSamplePDF(thread Light& light);

LightSample sampleAreaLight(thread Light& areaLight, device LightTriangle *lightTriangles, float3 r3);

LightSample sampleLight(float3 position,
                        thread Light& light,
                        device LightTriangle *lightTriangles,
                        texture2d<float> environmentMapTexture,
                        device float *environmentMapCDF,
                        float3 r3
                        );

Light selectLight(device Light *lights, device LightTriangle *lightTriangles, constant Uniforms& uniforms, float r, thread float& selectionPDF);

#endif
