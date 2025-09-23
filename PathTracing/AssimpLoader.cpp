//
//  AssimpLoader.cpp
//  PathTracing
//
//  Created by on 9/18/25.
//

#include "AssimpLoader.hpp"

int determineBXDF(float metallic, float roughness, float opacity, float ior) {
    if (opacity < 1.0f || ior > 1.01f) {
        return SPECULAR_TRANSMISSION;
    } else if (metallic > 0.5f) {
        return CONDUCTOR;
    } else {
        return DIFFUSE;
    }
}

EmbeddedTexture* extractEmbeddedTexture(const aiScene* scene, std::string path) {
    if (path.empty() || path[0] != '*') {
        printf("Not an embedded texture: %s\n", path.c_str());
        return nullptr;
    }
    
    int index = std::stoi(path.substr(1));
    if (index < 0 || index >= scene->mNumTextures) return nullptr;
    
    aiTexture* aiTex = scene->mTextures[index];
    if (!aiTex) return nullptr;

    EmbeddedTexture* tex = new EmbeddedTexture;
    tex->index = index;
    
    if (aiTex->mHeight == 0) {
        tex->isCompressed = true;
        tex->data = reinterpret_cast<unsigned char*>(aiTex->pcData);
        tex->dataSize = aiTex->mWidth;
        tex->width = 0;
        tex->height = 0;
        tex->formatHint = aiTex->achFormatHint;

    } else {
        tex->isCompressed = false;
        tex->data = reinterpret_cast<unsigned char*>(aiTex->pcData);
        tex->width = aiTex->mWidth;
        tex->height = aiTex->mHeight;
        tex->dataSize = aiTex->mWidth * aiTex->mHeight * sizeof(aiTexel);
    }
    
    return tex;
}

SceneData* loadModel(const char* path) {
    Assimp::Importer* importer = new Assimp::Importer();
    
    unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_GenUVCoords |
        aiProcess_JoinIdenticalVertices |
        aiProcess_OptimizeMeshes |
        aiProcess_ImproveCacheLocality |
        aiProcess_RemoveRedundantMaterials |
        aiProcess_FindDegenerates |
        aiProcess_FindInvalidData |
        aiProcess_FixInfacingNormals |
        aiProcess_FlipUVs |
        aiProcess_PreTransformVertices |
        aiProcess_EmbedTextures |
        aiProcess_ValidateDataStructure;
    
    const aiScene* scene = importer->ReadFile(path, flags);
    
    if (!scene || !scene->mRootNode) {
        printf("Failed to load model: %s\n", importer->GetErrorString());
        delete importer;
        return nullptr;
    }
    
    if (scene->HasTextures()) {
        printf("File has %d embedded textures\n", scene->mNumTextures);
    }
        
    std::string modelPath(path);
    size_t lastSlash = modelPath.find_last_of("/\\");
    std::string modelDir = (lastSlash != std::string::npos) ? modelPath.substr(0, lastSlash + 1) : "";
    
    SceneData* sceneData = new SceneData;
    sceneData->meshCount = scene->mNumMeshes;
    sceneData->meshes = new MeshData[sceneData->meshCount];
    printf("Loading model with %d meshes from: %s\n", scene->mNumMeshes, path);
    
    for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
        printf("Mesh: %d\n", m);
        aiMesh* aiMesh = scene->mMeshes[m];
        MeshData& mesh = sceneData->meshes[m];
        
        mesh.embeddedColorTexture = nullptr;
        mesh.embeddedRoughnessTexture = nullptr;
        mesh.embeddedMetallicTexture = nullptr;
        
        // Allocate vertex data
        mesh.vertexCount = aiMesh->mNumVertices;
        mesh.positions = new simd::float3[mesh.vertexCount];
        mesh.normals = new simd::float3[mesh.vertexCount];
        mesh.texCoords = new simd::float2[mesh.vertexCount];
        
        // Copy vertex data
        for (unsigned int v = 0; v < aiMesh->mNumVertices; v++) {
            mesh.positions[v].x = aiMesh->mVertices[v].x;
            mesh.positions[v].y = aiMesh->mVertices[v].y;
            mesh.positions[v].z = aiMesh->mVertices[v].z;
            
            if (aiMesh->HasNormals()) {
                mesh.normals[v].x = aiMesh->mNormals[v].x;
                mesh.normals[v].y = aiMesh->mNormals[v].y;
                mesh.normals[v].z = aiMesh->mNormals[v].z;
            } else {
                mesh.normals[v] = {0.0f, 1.0f, 0.0f};
            }
            
            if (aiMesh->mTextureCoords[0]) {
                mesh.texCoords[v].x = aiMesh->mTextureCoords[0][v].x;
                mesh.texCoords[v].y = aiMesh->mTextureCoords[0][v].y;
            } else {
                mesh.texCoords[v] = simd::float2(0.0f);
            }
        }
        
        // Copy indices
        mesh.indexCount = 0;
        for (unsigned int f = 0; f < aiMesh->mNumFaces; f++) {
            mesh.indexCount += aiMesh->mFaces[f].mNumIndices;
        }
        
        mesh.indices = new unsigned int[mesh.indexCount];
        unsigned int idx = 0;
        for (unsigned int f = 0; f < aiMesh->mNumFaces; f++) {
            aiFace& face = aiMesh->mFaces[f];
            for (unsigned int i = 0; i < face.mNumIndices; i++) {
                mesh.indices[idx++] = face.mIndices[i];
            }
        }
        
        // Initialize material with defaults
        
        mesh.material.color.value = simd::float3(0.8f);
        mesh.material.color.textureIndex = -1;
        
        mesh.material.refraction.value = 1.5f;
        mesh.material.refraction.textureIndex = -1;
        
        mesh.material.roughness.value = 0.5f;
        mesh.material.roughness.textureIndex = -1;
        
        mesh.material.metallic.value = 0.0f;
        mesh.material.metallic.textureIndex = -1;
        
        mesh.material.BXDFs = DIFFUSE;
        

        if (scene->mMaterials && aiMesh->mMaterialIndex < scene->mNumMaterials) {
            aiMaterial* mat = scene->mMaterials[aiMesh->mMaterialIndex];
            
            float opacity = 1.0f;
            mat->Get(AI_MATKEY_OPACITY, opacity);

            aiString alphaMode;
            if (mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS) {
                printf("Alpha mode: %s\n", alphaMode.C_Str());
                
                if (strcmp(alphaMode.C_Str(), "OPAQUE") == 0) {
                    opacity = 1.0f;
                } else if (strcmp(alphaMode.C_Str(), "BLEND") == 0) {
                    
                } else if (strcmp(alphaMode.C_Str(), "MASK") == 0) {
                    float cutoff;
                    mat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, cutoff);
                }
            }


            aiColor4D color;
            
            if (mat->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS) {
                mesh.material.color.value.x = color.r;
                mesh.material.color.value.y = color.g;
                mesh.material.color.value.z = color.b;
            } else if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
                mesh.material.color.value.x = color.r;
                mesh.material.color.value.y = color.g;
                mesh.material.color.value.z = color.b;
            }
            
            float ior = (opacity < 1.0f) ? 1.5f : 1.0f;
            mat->Get(AI_MATKEY_REFRACTI, ior);
            mesh.material.refraction.value = ior;

            float roughness = 0.5f;
            if (mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) != AI_SUCCESS) {
                float shininess = 0.0f;
                if (mat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
                    roughness = std::sqrt(2.0f / (shininess + 2.0f));
                }
            }
            mesh.material.roughness.value = roughness;
            
            float metallic = 0.0f;
            if (mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) != AI_SUCCESS) {
                float reflectivity = 0.0f;
                if (mat->Get(AI_MATKEY_REFLECTIVITY, reflectivity) == AI_SUCCESS) {
                    metallic = reflectivity > 0.5f ? 1.0f : 0.0f;
                }
            }
            mesh.material.metallic.value = metallic;
            
            mesh.material.BXDFs = determineBXDF(metallic, roughness, opacity, ior);
            aiString texPath;
                 
            if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS) {
                mesh.embeddedColorTexture = extractEmbeddedTexture(scene, texPath.C_Str());
                printf("Found embedded color texture: %s\n", texPath.C_Str());
            } else if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
                mesh.embeddedColorTexture = extractEmbeddedTexture(scene, texPath.C_Str());
                printf("Found embedded diffuse texture: %s\n", texPath.C_Str());
            }
            
            if (mat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &texPath) == AI_SUCCESS) {
                mesh.embeddedRoughnessTexture = extractEmbeddedTexture(scene, texPath.C_Str());
                printf("Found embedded roughness texture: %s\n", texPath.C_Str());
            }
            
            if (mat->GetTexture(aiTextureType_METALNESS, 0, &texPath) == AI_SUCCESS) {
                mesh.embeddedMetallicTexture = extractEmbeddedTexture(scene, texPath.C_Str());
                printf("Found embedded metallic texture: %s\n", texPath.C_Str());
            }
        }
    }
    
    printf("Successfully loaded %d meshes\n", scene->mNumMeshes);
    return sceneData;
}

void freeSceneData(SceneData* scene) {
    if (!scene) return;
    
    for (unsigned int i = 0; i < scene->meshCount; i++) {
        if (scene->meshes[i].embeddedColorTexture) {
            delete scene->meshes[i].embeddedColorTexture;
        }
        if (scene->meshes[i].embeddedRoughnessTexture) {
            delete scene->meshes[i].embeddedRoughnessTexture;
        }
        if (scene->meshes[i].embeddedMetallicTexture) {
            delete scene->meshes[i].embeddedMetallicTexture;
        }
        
        delete[] scene->meshes[i].positions;
        delete[] scene->meshes[i].normals;
        delete[] scene->meshes[i].texCoords;
        delete[] scene->meshes[i].indices;
    }
    
    delete[] scene->meshes;
    delete scene;
}
