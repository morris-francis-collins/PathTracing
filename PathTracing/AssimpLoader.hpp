//
//  AssimpLoader.hpp
//  PathTracing
//
//  Created by on 9/18/25.
//

#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/pbrmaterial.h>
#include <vector>
#include <string>
#include <cstdio>
#include <simd/simd.h>
#include "Metal/Materials.h"

struct EmbeddedTexture {
    const unsigned char* data;
    unsigned int dataSize;
    unsigned int width;
    unsigned int height;
    bool isCompressed;
    std::string formatHint;  // png, jpg, etc
    unsigned int index;
};

struct MeshData {
    simd::float3* positions;
    simd::float3* normals;
    simd::float2* texCoords;
    unsigned int* indices;
    
    unsigned int vertexCount;
    unsigned int indexCount;
    
    Material material;
    
    EmbeddedTexture* embeddedColorTexture = nullptr;
    EmbeddedTexture* embeddedRoughnessTexture = nullptr;
    EmbeddedTexture* embeddedMetallicTexture = nullptr;
};

struct SceneData {
    MeshData* meshes;
    unsigned int meshCount;
};

SceneData* loadModel(const char* path);
void freeSceneData(SceneData* scene);
