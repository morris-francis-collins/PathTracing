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

enum VertexType : unsigned int {
    CAMERA_VERTEX = 0,
    LIGHT_VERTEX = 1,
    SURFACE_VERTEX = 2
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

struct PathVertex {
    enum VertexType type;
    float3 throughput;
    SurfaceInteraction interaction;
    float forwardPDF;
    float reversePDF;
    bool delta;
    
//    PathVertex(VertexType _type, float3 _throughput, SurfaceInteraction _interaction, float _forwardPDF, float _reversePDF, bool _delta) {
//        type = _type;
//        throughput = _throughput;
//        interaction = _interaction;
//        forwardPDF = _forwardPDF;
//        reversePDF = _reversePDF;
//        delta = _delta;
//    }
    
    
};

#endif
