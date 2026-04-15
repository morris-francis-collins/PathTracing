//
//  GLTFLoader.cpp
//  PathTracing
//
//  Created on 3/22/25.
//

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "GLTFLoader.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static simd_float4x4 matrixFromCGLTF(cgltf_node* node) {
    float m[16];
    cgltf_node_transform_world(node, m);
    return *(simd_float4x4*)m;
}

static simd_float3x3 normalMatrixFrom4x4(simd_float4x4 m) {
    simd_float3x3 upper = {
        simd_make_float3(m.columns[0]),
        simd_make_float3(m.columns[1]),
        simd_make_float3(m.columns[2])
    };
    return simd_transpose(simd_inverse(upper));
}

static void loadImages(cgltf_data* data, std::vector<GLTFImage>& outImages, const std::string& baseDir) {
    outImages.resize(data->images_count);

    for (cgltf_size i = 0; i < data->images_count; i++) {
        cgltf_image& img = data->images[i];
        GLTFImage& out = outImages[i];
        out.data = nullptr;
        out.width = 0;
        out.height = 0;
        out.usage = GLTFImageUsage::Unknown;

        int w = 0, h = 0, channels = 0;
        unsigned char* pixels = nullptr;

        if (img.buffer_view) {
            const unsigned char* buf = static_cast<const unsigned char*>(img.buffer_view->buffer->data) + img.buffer_view->offset;
            int bufSize = static_cast<int>(img.buffer_view->size);
            pixels = stbi_load_from_memory(buf, bufSize, &w, &h, &channels, 4);
        } else if (img.uri && strncmp(img.uri, "data:", 5) != 0) {
            std::string fullPath = baseDir + img.uri;
            pixels = stbi_load(fullPath.c_str(), &w, &h, &channels, 4);
        }

        if (pixels) {
            out.data = pixels;
            out.width = (uint32_t)w;
            out.height = (uint32_t)h;
        } else {
            printf("[GLTFLoader] Warning: failed to decode image %zu\n", i);
        }
    }
}

static int32_t resolveTextureIndex(cgltf_data* data, cgltf_texture* tex) {
    if (!tex || !tex->image)
        return -1;
    return (int32_t)(tex->image - data->images);
}

static void tagImageUsage(std::vector<GLTFImage>& images, int32_t index, GLTFImageUsage usage) {
    if (index >= 0 && index < (int32_t)images.size()) {
        if (images[index].usage == GLTFImageUsage::Unknown) {
            images[index].usage = usage;
        }
    }
}

static void loadMaterials(cgltf_data* data, std::vector<Material>& outMaterials, std::vector<GLTFImage>& images) {
    outMaterials.resize(data->materials_count);

    for (cgltf_size i = 0; i < data->materials_count; i++) {
        cgltf_material& mat = data->materials[i];
        Material& out = outMaterials[i];
        memset(&out, 0, sizeof(out));

        out.colorTextureIndex = -1;
        out.roughnessTextureIndex = -1;
        out.metallicTextureIndex = -1;
        out.emissionTextureIndex = -1;
        out.transmissionTextureIndex = -1;
        out.normalTextureIndex = -1;
        out.clearcoatTextureIndex = -1;
        out.clearcoatRoughnessTextureIndex = -1;
        out.clearcoatNormalTextureIndex = -1;
        out.anisotropyTextureIndex = -1;
        out.specularTextureIndex = -1;
        out.specularColorTextureIndex = -1;
        out.thicknessTextureIndex = -1;
        out.diffuseTransmissionTextureIndex = -1;
        out.diffuseTransmissionColorTextureIndex = -1;
        out.sheenColorTextureIndex = -1;
        out.sheenRoughnessTextureIndex = -1;
        out.iridescenceTextureIndex = -1;
        out.iridescenceThicknessTextureIndex = -1;

        out.emissiveStrength = 1.0f;
        out.ior = 1.5f;
        out.alphaMode = 0;
        out.alphaCutoff = 0.5f;
        out.attenuationColor = simd_make_float3(1.0f, 1.0f, 1.0f);
        out.attenuationDistance = INFINITY;
        out.normalScale = 1.0f;
        out.doubleSided = 0;
        out.specularFactor = 1.0f;
        out.specularColor = simd_make_float3(1.0f, 1.0f, 1.0f);
        out.diffuseTransmissionFactor = 0.0f;
        out.diffuseTransmissionColor = simd_make_float3(1.0f, 1.0f, 1.0f);
        out.iridescenceFactor = 0.0f;
        out.iridescenceIor = 1.3f;
        out.iridescenceThicknessMin = 100.0f;
        out.iridescenceThicknessMax = 400.0f;
        out.dispersion = 0.0f;

        if (mat.has_pbr_metallic_roughness) {
            cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;

            out.colorValue = simd_make_float3(
                pbr.base_color_factor[0],
                pbr.base_color_factor[1],
                pbr.base_color_factor[2]
            );

            int32_t baseColorIdx = resolveTextureIndex(data, pbr.base_color_texture.texture);
            out.colorTextureIndex = baseColorIdx;
            tagImageUsage(images, baseColorIdx, GLTFImageUsage::BaseColor);

            out.metallicValue = pbr.metallic_factor;
            out.roughnessValue = pbr.roughness_factor;

            int32_t mrIdx = resolveTextureIndex(data, pbr.metallic_roughness_texture.texture);
            out.metallicTextureIndex = mrIdx;
            out.roughnessTextureIndex = mrIdx;
            tagImageUsage(images, mrIdx, GLTFImageUsage::MetallicRoughness);
        } else {
            out.colorValue = simd_make_float3(0.8f, 0.8f, 0.8f);
            out.roughnessValue = 0.5f;
            out.metallicValue = 0.0f;
        }

        int32_t normalIdx = resolveTextureIndex(data, mat.normal_texture.texture);
        out.normalTextureIndex = normalIdx;
        out.normalScale = (mat.normal_texture.scale != 0.0f) ? mat.normal_texture.scale : 1.0f;
        tagImageUsage(images, normalIdx, GLTFImageUsage::Normal);

        out.emissionValue = simd_make_float3(
            mat.emissive_factor[0],
            mat.emissive_factor[1],
            mat.emissive_factor[2]
        );
        int32_t emIdx = resolveTextureIndex(data, mat.emissive_texture.texture);
        out.emissionTextureIndex = emIdx;
        tagImageUsage(images, emIdx, GLTFImageUsage::Emissive);

        if (mat.has_emissive_strength) {
            out.emissiveStrength = mat.emissive_strength.emissive_strength;
        }

        switch (mat.alpha_mode) {
            case cgltf_alpha_mode_opaque: out.alphaMode = 0; break;
            case cgltf_alpha_mode_mask:   out.alphaMode = 1; break;
            case cgltf_alpha_mode_blend:  out.alphaMode = 2; break;
            default: printf("[GLTFLoader] Warning: Alpha mode not found\n");
        }
        out.alphaCutoff = mat.alpha_cutoff;
        out.doubleSided = mat.double_sided ? 1 : 0;

        if (mat.has_ior) {
            out.ior = mat.ior.ior;
        }

        if (mat.has_specular) {
            out.specularFactor = mat.specular.specular_factor;
            out.specularColor = simd_make_float3(
                mat.specular.specular_color_factor[0],
                mat.specular.specular_color_factor[1],
                mat.specular.specular_color_factor[2]
            );

            int32_t specIdx = resolveTextureIndex(data, mat.specular.specular_texture.texture);
            out.specularTextureIndex = specIdx;
            tagImageUsage(images, specIdx, GLTFImageUsage::Linear);

            int32_t specColorIdx = resolveTextureIndex(data, mat.specular.specular_color_texture.texture);
            out.specularColorTextureIndex = specColorIdx;
            tagImageUsage(images, specColorIdx, GLTFImageUsage::BaseColor);
        }

        if (mat.has_transmission) {
            out.transmissionValue = mat.transmission.transmission_factor;
            int32_t transIdx = resolveTextureIndex(data, mat.transmission.transmission_texture.texture);
            out.transmissionTextureIndex = transIdx;
            tagImageUsage(images, transIdx, GLTFImageUsage::Transmission);
        }

        if (mat.has_volume) {
            out.thicknessFactor = mat.volume.thickness_factor;
            out.attenuationColor = simd_make_float3(
                mat.volume.attenuation_color[0],
                mat.volume.attenuation_color[1],
                mat.volume.attenuation_color[2]
            );
            out.attenuationDistance = mat.volume.attenuation_distance;

            int32_t thickIdx = resolveTextureIndex(data, mat.volume.thickness_texture.texture);
            out.thicknessTextureIndex = thickIdx;
            tagImageUsage(images, thickIdx, GLTFImageUsage::Linear);
        }

        if (mat.has_diffuse_transmission) {
            out.diffuseTransmissionFactor = mat.diffuse_transmission.diffuse_transmission_factor;
            out.diffuseTransmissionColor = simd_make_float3(
                mat.diffuse_transmission.diffuse_transmission_color_factor[0],
                mat.diffuse_transmission.diffuse_transmission_color_factor[1],
                mat.diffuse_transmission.diffuse_transmission_color_factor[2]
            );

            int32_t dtIdx = resolveTextureIndex(data, mat.diffuse_transmission.diffuse_transmission_texture.texture);
            out.diffuseTransmissionTextureIndex = dtIdx;
            tagImageUsage(images, dtIdx, GLTFImageUsage::Linear);

            int32_t dtcIdx = resolveTextureIndex(data, mat.diffuse_transmission.diffuse_transmission_color_texture.texture);
            out.diffuseTransmissionColorTextureIndex = dtcIdx;
            tagImageUsage(images, dtcIdx, GLTFImageUsage::BaseColor);
        }

        if (mat.has_clearcoat) {
            out.clearcoatValue = mat.clearcoat.clearcoat_factor;
            out.clearcoatRoughnessValue = mat.clearcoat.clearcoat_roughness_factor;

            int32_t ccIdx = resolveTextureIndex(data, mat.clearcoat.clearcoat_texture.texture);
            out.clearcoatTextureIndex = ccIdx;
            tagImageUsage(images, ccIdx, GLTFImageUsage::Clearcoat);

            int32_t ccrIdx = resolveTextureIndex(data, mat.clearcoat.clearcoat_roughness_texture.texture);
            out.clearcoatRoughnessTextureIndex = ccrIdx;
            tagImageUsage(images, ccrIdx, GLTFImageUsage::Clearcoat);

            int32_t ccnIdx = resolveTextureIndex(data, mat.clearcoat.clearcoat_normal_texture.texture);
            out.clearcoatNormalTextureIndex = ccnIdx;
            tagImageUsage(images, ccnIdx, GLTFImageUsage::Normal);
        }

        if (mat.has_anisotropy) {
            out.anisotropyStrength = mat.anisotropy.anisotropy_strength;
            out.anisotropyRotation = mat.anisotropy.anisotropy_rotation;
            int32_t aniIdx = resolveTextureIndex(data, mat.anisotropy.anisotropy_texture.texture);
            out.anisotropyTextureIndex = aniIdx;
            tagImageUsage(images, aniIdx, GLTFImageUsage::Linear);
        }

        if (mat.has_sheen) {
            out.sheenColor = simd_make_float3(
                mat.sheen.sheen_color_factor[0],
                mat.sheen.sheen_color_factor[1],
                mat.sheen.sheen_color_factor[2]
            );
            out.sheenRoughness = mat.sheen.sheen_roughness_factor;

            int32_t scIdx = resolveTextureIndex(data, mat.sheen.sheen_color_texture.texture);
            out.sheenColorTextureIndex = scIdx;
            tagImageUsage(images, scIdx, GLTFImageUsage::BaseColor);

            int32_t srIdx = resolveTextureIndex(data, mat.sheen.sheen_roughness_texture.texture);
            out.sheenRoughnessTextureIndex = srIdx;
            tagImageUsage(images, srIdx, GLTFImageUsage::Linear);
        }

        if (mat.has_iridescence) {
            out.iridescenceFactor = mat.iridescence.iridescence_factor;
            out.iridescenceIor = mat.iridescence.iridescence_ior;
            out.iridescenceThicknessMin = mat.iridescence.iridescence_thickness_min;
            out.iridescenceThicknessMax = mat.iridescence.iridescence_thickness_max;

            int32_t irIdx = resolveTextureIndex(data, mat.iridescence.iridescence_texture.texture);
            out.iridescenceTextureIndex = irIdx;
            tagImageUsage(images, irIdx, GLTFImageUsage::Linear);

            int32_t irtIdx = resolveTextureIndex(data, mat.iridescence.iridescence_thickness_texture.texture);
            out.iridescenceThicknessTextureIndex = irtIdx;
            tagImageUsage(images, irtIdx, GLTFImageUsage::Linear);
        }

        if (mat.has_dispersion) {
            out.dispersion = mat.dispersion.dispersion;
        }
    }

    if (outMaterials.empty()) {
        Material def;
        memset(&def, 0, sizeof(def));
        def.colorValue = simd_make_float3(0.8f, 0.8f, 0.8f);
        def.colorTextureIndex = -1;
        def.roughnessTextureIndex = -1;
        def.metallicTextureIndex = -1;
        def.emissionTextureIndex = -1;
        def.emissiveStrength = 1.0f;
        def.ior = 1.5f;
        def.alphaCutoff = 0.5f;
        def.specularFactor = 1.0f;
        def.specularColor = simd_make_float3(1, 1, 1);
        def.specularTextureIndex = -1;
        def.specularColorTextureIndex = -1;
        def.transmissionTextureIndex = -1;
        def.thicknessTextureIndex = -1;
        def.attenuationColor = simd_make_float3(1, 1, 1);
        def.attenuationDistance = INFINITY;
        def.diffuseTransmissionTextureIndex = -1;
        def.diffuseTransmissionColor = simd_make_float3(1, 1, 1);
        def.diffuseTransmissionColorTextureIndex = -1;
        def.normalTextureIndex = -1;
        def.normalScale = 1.0f;
        def.clearcoatTextureIndex = -1;
        def.clearcoatRoughnessTextureIndex = -1;
        def.clearcoatNormalTextureIndex = -1;
        def.anisotropyTextureIndex = -1;
        def.sheenColorTextureIndex = -1;
        def.sheenRoughnessTextureIndex = -1;
        def.iridescenceTextureIndex = -1;
        def.iridescenceThicknessTextureIndex = -1;
        def.iridescenceIor = 1.3f;
        def.iridescenceThicknessMin = 100.0f;
        def.iridescenceThicknessMax = 400.0f;
        outMaterials.push_back(def);
    }
}

static void processMesh(cgltf_data* data, cgltf_mesh* mesh, simd_float4x4 worldTransform, std::vector<GLTFPrimitive>& outPrimitives) {
    simd_float3x3 normalMatrix = normalMatrixFrom4x4(worldTransform);

    simd_float3x3 upper3x3 = {
        simd_make_float3(worldTransform.columns[0]),
        simd_make_float3(worldTransform.columns[1]),
        simd_make_float3(worldTransform.columns[2])
    };

    for (cgltf_size p = 0; p < mesh->primitives_count; p++) {
        cgltf_primitive& prim = mesh->primitives[p];

        if (prim.type != cgltf_primitive_type_triangles) {
            printf("[GLTFLoader] Skipping non-triangle primitive (type %d)\n", prim.type);
            continue;
        }

        cgltf_accessor* posAcc = nullptr;
        cgltf_accessor* normAcc = nullptr;
        cgltf_accessor* uvAcc = nullptr;
        cgltf_accessor* tangentAcc = nullptr;

        for (cgltf_size a = 0; a < prim.attributes_count; a++) {
            cgltf_attribute& attr = prim.attributes[a];
            switch (attr.type) {
                case cgltf_attribute_type_position:
                    posAcc = attr.data;
                    break;
                case cgltf_attribute_type_normal:
                    normAcc = attr.data;
                    break;
                case cgltf_attribute_type_texcoord:
                    if (attr.index == 0) uvAcc = attr.data;
                    break;
                case cgltf_attribute_type_tangent:
                    tangentAcc = attr.data;
                    break;
                default:
                    break;
            }
        }

        if (!posAcc) {
            printf("[GLTFLoader] Primitive has no POSITION attribute, skipping\n");
            continue;
        }

        uint32_t matIdx = 0;
        if (prim.material) {
            matIdx = (uint32_t)(prim.material - data->materials);
        }

        cgltf_size triCount;
        if (prim.indices) {
            triCount = prim.indices->count / 3;
        } else {
            triCount = posAcc->count / 3;
        }

        uint32_t vertexCount = (uint32_t)(triCount * 3);
        GLTFVertex* verts = new GLTFVertex[vertexCount];

        for (cgltf_size tri = 0; tri < triCount; tri++) {
            for (int v = 0; v < 3; v++) {
                cgltf_size srcIdx;
                if (prim.indices) {
                    srcIdx = cgltf_accessor_read_index(prim.indices, tri * 3 + v);
                } else {
                    srcIdx = tri * 3 + v;
                }

                GLTFVertex& vert = verts[tri * 3 + v];

                float pos[3] = {0, 0, 0};
                cgltf_accessor_read_float(posAcc, srcIdx, pos, 3);
                simd_float4 worldPos = simd_mul(worldTransform,
                    simd_make_float4(pos[0], pos[1], pos[2], 1.0f));
                vert.position = simd_make_float3(worldPos.x, worldPos.y, worldPos.z);

                if (normAcc) {
                    float n[3] = {0, 1, 0};
                    cgltf_accessor_read_float(normAcc, srcIdx, n, 3);
                    simd_float3 wn = simd_mul(normalMatrix,
                        simd_make_float3(n[0], n[1], n[2]));
                    float len = simd_length(wn);
                    vert.normal = (len > 1e-8f) ? (wn / len) : simd_make_float3(0, 1, 0);
                } else {
                    vert.normal = simd_make_float3(0.0f, 1.0f, 0.0f);
                }

                if (uvAcc) {
                    float uv[2] = {0, 0};
                    cgltf_accessor_read_float(uvAcc, srcIdx, uv, 2);
                    vert.texCoord = simd_make_float2(uv[0], uv[1]);
                } else {
                    vert.texCoord = simd_make_float2(0.0f, 0.0f);
                }

                if (tangentAcc) {
                    float t[4] = {1, 0, 0, 1};
                    cgltf_accessor_read_float(tangentAcc, srcIdx, t, 4);
                    simd_float3 wt = simd_normalize(simd_mul(upper3x3,
                        simd_make_float3(t[0], t[1], t[2])));
                    vert.tangent = simd_make_float4(wt.x, wt.y, wt.z, t[3]);
                } else {
                    vert.tangent = simd_make_float4(1.0f, 0.0f, 0.0f, 1.0f);
                }
            }
        }

        GLTFPrimitive outPrim;
        outPrim.vertices = verts;
        outPrim.vertexCount = vertexCount;
        outPrim.materialIndex = matIdx;
        outPrimitives.push_back(outPrim);
    }
}

static void processNode(cgltf_data* data, cgltf_node* node, std::vector<GLTFPrimitive>& outPrimitives) {
    if (node->mesh) {
        simd_float4x4 world = matrixFromCGLTF(node);
        processMesh(data, node->mesh, world, outPrimitives);
    }

    for (cgltf_size i = 0; i < node->children_count; i++) {
        processNode(data, node->children[i], outPrimitives);
    }
}

GLTFSceneData* loadGLTFScene(const char* path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, path, &data);
    if (result != cgltf_result_success) {
        printf("[GLTFLoader] Failed to parse: %s (error %d)\n", path, (int)result);
        return nullptr;
    }

    result = cgltf_load_buffers(&options, data, path);
    if (result != cgltf_result_success) {
        printf("[GLTFLoader] Failed to load buffers: %s (error %d)\n", path, (int)result);
        cgltf_free(data);
        return nullptr;
    }

    result = cgltf_validate(data);
    if (result != cgltf_result_success) {
        printf("[GLTFLoader] Validation warnings for: %s (error %d)\n", path, (int)result);
    }

    std::string modelPath(path);
    size_t lastSlash = modelPath.find_last_of("/\\");
    std::string baseDir = (lastSlash != std::string::npos) ? modelPath.substr(0, lastSlash + 1) : "";

    std::vector<GLTFImage> images;
    loadImages(data, images, baseDir);

    std::vector<Material> materials;
    loadMaterials(data, materials, images);

    std::vector<GLTFPrimitive> primitives;
    if (data->scenes_count > 0) {
        cgltf_scene* scene = data->scene ? data->scene : &data->scenes[0];
        for (cgltf_size n = 0; n < scene->nodes_count; n++) {
            processNode(data, scene->nodes[n], primitives);
        }
    } else {
        for (cgltf_size n = 0; n < data->nodes_count; n++) {
            if (data->nodes[n].parent == nullptr) {
                processNode(data, &data->nodes[n], primitives);
            }
        }
    }

    printf("[GLTFLoader] Loaded %zu primitives, %zu materials, %zu images from %s\n",
           primitives.size(), materials.size(), images.size(), path);

    GLTFSceneData* scene = new GLTFSceneData;

    scene->primitiveCount = (uint32_t)primitives.size();
    scene->primitives = new GLTFPrimitive[scene->primitiveCount];
    memcpy(scene->primitives, primitives.data(),
           primitives.size() * sizeof(GLTFPrimitive));

    scene->materialCount = (uint32_t)materials.size();
    scene->materials = new Material[scene->materialCount];
    memcpy(scene->materials, materials.data(),
           materials.size() * sizeof(Material));

    scene->imageCount = (uint32_t)images.size();
    scene->images = new GLTFImage[scene->imageCount];
    memcpy(scene->images, images.data(),
           images.size() * sizeof(GLTFImage));

    cgltf_free(data);
    return scene;
}

void freeGLTFScene(GLTFSceneData* scene) {
    if (!scene) return;

    for (uint32_t i = 0; i < scene->primitiveCount; i++) {
        delete[] scene->primitives[i].vertices;
    }
    
    delete[] scene->primitives;
    delete[] scene->materials;

    for (uint32_t i = 0; i < scene->imageCount; i++) {
        if (scene->images[i].data) {
            stbi_image_free(scene->images[i].data);
        }
    }
    delete[] scene->images;

    delete scene;
}
