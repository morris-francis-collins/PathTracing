//
//  Lights.h
//  PathTracing
//
//  Created on 7/19/25.
//

#pragma once

#define MAX_AREA_LIGHTS 16

#include <simd/simd.h>
#include "Utility.h"

enum LightType : unsigned int {
    POINT_LIGHT = 0,
    AREA_LIGHT = 1,
};

struct Light {
    enum LightType type;
    unsigned int index;
    int delta;
    vector_float3 position;
    vector_float3 color;
    unsigned int firstTriangleIndex; // area lights only below
    unsigned int triangleCount;
    float totalArea;
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

struct AreaLight {
    vector_float3 position;
    vector_float3 color;
    unsigned int firstTriangleIndex;
    unsigned int triangleCount;
    float totalArea;
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

struct LightSample {
    float3 position;
    float3 normal;
    float3 emission;
    float PDF;
};

LightSample sampleLight(thread AreaLight& areaLight, device LightTriangle *lightTriangles, float3 r3);

AreaLight selectLight(device AreaLight *areaLights, device LightTriangle *lightTriangles, constant Uniforms& uniforms, float r, thread float& selectionPDF);

#endif
