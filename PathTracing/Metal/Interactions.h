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

struct PrimitiveData {
    vector_float3 n0, n1, n2;
    vector_float2 uv0, uv1, uv2;
    int materialIndex;
    int primitiveLightIndex;
};

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

struct IntersectionResult {
    const device void* primitive_data;
    float2 triangle_barycentric_coord;
    float distance;
    uint instance_id;
    intersection_type type;
};

SurfaceInteraction getSurfaceInteraction(ray ray,
                                         IntersectionResult intersection,
                                         device MTLAccelerationStructureInstanceDescriptor *instances,
                                         instance_acceleration_structure accelerationStructure,
                                         constant int* instanceLightIndices,
                                         constant Textures* textures,
                                         constant Material* materials
                                         );

template <bool acceptAnyIntersection>
IntersectionResult intersect(ray ray, instance_acceleration_structure accelerationStructure) {
    intersector<triangle_data, instancing> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    i.accept_any_intersection(acceptAnyIntersection);

    auto result = i.intersect(ray, accelerationStructure, __UINT32_MAX__);

    IntersectionResult intersection;
    intersection.triangle_barycentric_coord = result.triangle_barycentric_coord;
    intersection.instance_id                = result.instance_id;
    intersection.primitive_data             = result.primitive_data;
    intersection.distance                   = result.distance;
    intersection.type                       = result.type;

    return intersection;
}

inline float3 transformPoint(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 1.0f);
}

inline float3 transformDirection(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 0.0f);
}

template <typename T>
inline T interpolateVertexAttribute(device T *attributes, unsigned int primitiveIndex, float2 uv)
{
    T T0 = attributes[primitiveIndex * 3 + 0];
    T T1 = attributes[primitiveIndex * 3 + 1];
    T T2 = attributes[primitiveIndex * 3 + 2];

    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

template <typename T>
inline T interpolateVertexAttribute(T v0, T v1, T v2, float2 uv) {
    return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
}

bool isVisible(float3 pos1, float3 normal1,
               float3 pos2, float3 normal2,
               device MTLAccelerationStructureInstanceDescriptor *instances,
               instance_acceleration_structure accelerationStructure);

#endif
