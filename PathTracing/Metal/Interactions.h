//
//  Interactions.h
//  PathTracing
//
//  Created on 7/19/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Materials.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

struct Textures {
    array<texture2d<float>, MAX_TEXTURES> textures;
};

struct SurfaceInteraction {
    float3 position;
    float3 normal;
    SampledMaterial material;
    int lightIndex;
    float3 emission;
    
    SurfaceInteraction() {
        
    }
    
    SurfaceInteraction(float3 _position, float3 _normal, SampledMaterial _material, int _lightIndex, float3 _emission) {
        position = _position;
        normal = _normal;
        material = _material;
        lightIndex = _lightIndex;
        emission = _emission;
    }
    
    bool hitLight() {
        return lightIndex != -1;
    }
};

typedef intersector<triangle_data, instancing, world_space_data>::result_type IntersectionResult;

IntersectionResult intersect(ray ray,
                             unsigned int mask,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             instance_acceleration_structure accelerationStructure,
                             bool accept_any_intersection);

SurfaceInteraction getSurfaceInteraction(ray ray,
                                         IntersectionResult intersection,
                                         device void *resources,
                                         device MTLAccelerationStructureInstanceDescriptor *instances,
                                         instance_acceleration_structure accelerationStructure,
                                         constant int* instanceLightIndices,
                                         int resourcesStride,
                                         constant Textures* textures
                                         );

inline float3 transformPoint(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 1.0f);
}

inline float3 transformDirection(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 0.0f);
}

template<typename T>
inline T interpolateVertexAttribute(device T *attributes, unsigned int primitiveIndex, float2 uv)
{
    T T0 = attributes[primitiveIndex * 3 + 0];
    T T1 = attributes[primitiveIndex * 3 + 1];
    T T2 = attributes[primitiveIndex * 3 + 2];

    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

bool isVisible(float3 pos1, float3 normal1,
               float3 pos2, float3 normal2,
               device void *resources,
               device MTLAccelerationStructureInstanceDescriptor *instances,
               instance_acceleration_structure accelerationStructure);

#endif
