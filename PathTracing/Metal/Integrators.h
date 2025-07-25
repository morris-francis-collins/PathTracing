//
//  Integrators.h
//  PathTracing
//
//  Created by Moritz Kohlenz on 7/19/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"

#define MAX_PATH_LENGTH 20
#define MAX_CAMERA_PATH_LENGTH (MAX_PATH_LENGTH + 2)
#define MAX_LIGHT_PATH_LENGTH (MAX_PATH_LENGTH + 1)

#define CAMERA_VERTEX 1
#define LIGHT_VERTEX 2
#define SURFACE_VERTEX 4

struct PathVertex {
    vector_float3 position;
    vector_float3 normal;
    vector_float3 tangent;
    vector_float3 bitangent;
    vector_float3 throughput;
    vector_float3 material_color;
    vector_float3 incoming_direction;
    struct Material material;
    float mediumDistance;
    float forwardPDF;
    float reversePDF;
    vector_float3 BSDF;
    int is_delta;
    int in_medium;
    int type;
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
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
                      );
#endif
