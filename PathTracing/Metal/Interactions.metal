//
//  Interactions.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include "Interactions.h"

using namespace metal;
using namespace raytracing;

void sampleBXDF(thread SampledMaterial& material) {
    if (material.transmission > 0.01f) {
        material.BXDFs = DIELECTRIC_TRANSMISSION;
    } else if (material.metallic > 0.5f) {
        material.BXDFs = CONDUCTOR;
    } else if (material.ior > 1.01f) {
        material.BXDFs = DIELECTRIC_REFLECTION;
    } else {
        material.BXDFs = DIFFUSE;
    }
}

SampledMaterial sampleMaterial(Material material, float2 uv, constant Textures* textures) {
    SampledMaterial out;
    constexpr sampler s(address::repeat, filter::linear);
    
    float alpha = 1.0f;
    if (material.colorTextureIndex >= 0) {
        float4 tex = textures->textures[material.colorTextureIndex].sample(s, uv);
        out.color = material.colorValue * tex.rgb;
        alpha = tex.a;
    } else {
        out.color = material.colorValue;
    }

    if (material.roughnessTextureIndex >= 0) {
        out.roughness = material.roughnessValue * textures->textures[material.roughnessTextureIndex].sample(s, uv).g;
    } else {
        out.roughness = material.roughnessValue;
    }

    if (material.metallicTextureIndex >= 0) {
        out.metallic = material.metallicValue * textures->textures[material.metallicTextureIndex].sample(s, uv).b;
    } else {
        out.metallic = material.metallicValue;
    }

    if (material.emissionTextureIndex >= 0) {
        out.emission = material.emissionValue * textures->textures[material.emissionTextureIndex].sample(s, uv).rgb * material.emissiveStrength;
    } else {
        out.emission = material.emissionValue * material.emissiveStrength;
    }

    if (material.transmissionTextureIndex >= 0) {
        out.transmission = material.transmissionValue * textures->textures[material.transmissionTextureIndex].sample(s, uv).r;
    } else {
        out.transmission = material.transmissionValue;
    }

    if (material.clearcoatTextureIndex >= 0) {
        out.clearcoat = material.clearcoatValue * textures->textures[material.clearcoatTextureIndex].sample(s, uv).r;
    } else {
        out.clearcoat = material.clearcoatValue;
    }

    if (material.clearcoatRoughnessTextureIndex >= 0) {
        out.clearcoatRoughness = material.clearcoatRoughnessValue * textures->textures[material.clearcoatRoughnessTextureIndex].sample(s, uv).g;
    } else {
        out.clearcoatRoughness = material.clearcoatRoughnessValue;
    }

    out.ior = material.ior;
    out.thicknessFactor = material.thicknessFactor;
    out.attenuationColor = material.attenuationColor;
    out.attenuationDistance = material.attenuationDistance;

    out.alphaMode = material.alphaMode;
    out.alphaCutoff = material.alphaCutoff;
    out.alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0f : alpha;

    sampleBXDF(out);
    return out;
}

SurfaceInteraction getSurfaceInteraction(ray ray,
                                         IntersectionResult intersection,
                                         device MTLAccelerationStructureInstanceDescriptor *instances,
                                         instance_acceleration_structure accelerationStructure,
                                         constant int* instanceLightIndices,
                                         constant Textures* textures,
                                         constant Material* materials
                                         )
{
    SurfaceInteraction surfaceInteraction;
    surfaceInteraction.position = ray.origin + ray.direction * intersection.distance;
    
    unsigned int instanceIndex = intersection.instance_id;
    
    auto p = instances[instanceIndex].transformationMatrix;
    float4x3 objectToWorldTransform = float4x3(
        float3(p[0].x, p[0].y, p[0].z),
        float3(p[1].x, p[1].y, p[1].z),
        float3(p[2].x, p[2].y, p[2].z),
        float3(p[3].x, p[3].y, p[3].z)
    );
    
    float2 barycentric_coords = intersection.triangle_barycentric_coord;
    
    const device PrimitiveData* primitiveData = static_cast<const device PrimitiveData*>(intersection.primitive_data);

    float3 objectNormal = interpolateVertexAttribute(primitiveData->n0, primitiveData->n1, primitiveData->n2, barycentric_coords);
    float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldTransform));
    surfaceInteraction.normal = worldNormal;
    
    float2 uv = interpolateVertexAttribute(primitiveData->uv0, primitiveData->uv1, primitiveData->uv2, barycentric_coords);
    
    constant Material& material = materials[primitiveData->materialIndex];
    constexpr sampler textureSampler(address::repeat, filter::linear);
    int primitiveLightIndex = primitiveData->primitiveLightIndex;
    
    if (primitiveLightIndex != -1) {
        surfaceInteraction.lightIndex = instanceLightIndices[instanceIndex] + primitiveLightIndex;
        surfaceInteraction.emission = material.emissionValue * material.emissiveStrength;
        if (material.emissionTextureIndex >= 0) {
            surfaceInteraction.emission *= textures->textures[material.emissionTextureIndex].sample(textureSampler, uv).rgb;
        }
    } else {
        surfaceInteraction.lightIndex = -1;
        surfaceInteraction.emission = float3(0.0f);
    }
    
    if (material.normalTextureIndex >= 0) {
        float3 texNormal = textures->textures[material.normalTextureIndex].sample(textureSampler, uv).xyz;
        texNormal = texNormal * 2.0 - 1.0;
        texNormal.xy *= material.normalScale;
        
        float3 T, B;
        createOrthonormalBasis(surfaceInteraction.normal, T, B);

        surfaceInteraction.normal = normalize(T * texNormal.x + B * texNormal.y + surfaceInteraction.normal * texNormal.z);
    }

    surfaceInteraction.material = sampleMaterial(material, uv, textures);
    
    return surfaceInteraction;
}

bool isVisible(float3 pos1, float3 normal1,
               float3 pos2, float3 normal2,
               device MTLAccelerationStructureInstanceDescriptor *instances,
               instance_acceleration_structure accelerationStructure)
{
    float3 w = pos2 - pos1;
    float dist = length(w);
    w /= dist;

    float3 origin = pos1 + w * 1e-4f;
    float3 target = pos2 - w * 1e-4f;

    float3 dir = target - origin;
    float len = length(dir);
    dir /= len;

    ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = dir;
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = len - 1e-4f;
    
    IntersectionResult intersection = intersect<true>(shadowRay, accelerationStructure);
    return intersection.type == intersection_type::none;
}
