//
//  GLTFLoader.hpp
//  PathTracing
//
//  Created on 3/22/25.
//

#pragma once

#include <simd/simd.h>
#include <vector>
#include <cstdint>
#include "Materials.h"

struct cgltf_data;

enum class GLTFImageUsage : unsigned int {
    BaseColor = 0,
    MetallicRoughness,
    Normal,
    Emissive,
    Occlusion,
    Transmission,
    Clearcoat,
    Unknown
};

struct GLTFImage {
    unsigned char* data;
    uint32_t width;
    uint32_t height;
    GLTFImageUsage usage;
};

struct GLTFVertex {
    simd_float3 position;
    simd_float3 normal;
    simd_float2 texCoord;
    simd_float4 tangent;
};

struct GLTFPrimitive {
    GLTFVertex* vertices;
    uint32_t vertexCount;
    uint32_t materialIndex;
};

struct GLTFSceneData {
    GLTFPrimitive* primitives;
    uint32_t primitiveCount;

    Material* materials;
    uint32_t materialCount;

    GLTFImage* images;
    uint32_t imageCount;
};

GLTFSceneData* loadGLTFScene(const char* path);
void freeGLTFScene(GLTFSceneData* scene);
