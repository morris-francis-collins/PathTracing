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

struct SurfaceInteraction {
    float3 position;
    float3 normal;
    Material material;
    float3 textureColor;
    bool hitLight;
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
                                         int resourcesStride,
                                         array<texture2d<float>, MAX_TEXTURES> textureArray
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

inline bool isVisible(float3 pos1, float3 pos2,
                      float3 n1, float3 n2,
                      device void *resources,
                      device MTLAccelerationStructureInstanceDescriptor *instances,
                      instance_acceleration_structure accelerationStructure
                      )
{
    float epsilon = calculateEpsilon(pos1);
    float3 w = pos2 - pos1;
    float dist = length(w);
    w /= dist;

    pos1 += calculateOffset(w, n1, epsilon);
    pos2 += calculateOffset(-w, n2, epsilon);
    
    ray shadowRay;
    shadowRay.direction = w;
    shadowRay.origin = pos1 + calculateOffset(w, w, epsilon);
    shadowRay.min_distance = epsilon;
    shadowRay.max_distance = dist - epsilon;
    
    IntersectionResult intersection = intersect(
                                                shadowRay,
                                                RAY_MASK_SHADOW,
                                                resources,
                                                instances,
                                                accelerationStructure,
                                                true
                                                );
    
    return intersection.type == intersection_type::none;
}

#endif
