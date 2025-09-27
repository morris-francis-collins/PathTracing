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

struct TriangleResources {
    device float3 *vertexNormals;
//    device float3 *vertexColors;
    device Material *vertexMaterials;
    device float2 *vertexUVs;
    device int *vertexPrimitiveLightIndices;
};

IntersectionResult intersect(ray ray,
                             unsigned int mask,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             instance_acceleration_structure accelerationStructure,
                             bool accept_any_intersection)
{
    intersection_params params;
    
    intersection_query<triangle_data, instancing> i;
    
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);

    params.accept_any_intersection(accept_any_intersection); // get any, not just the closest

    i.reset(ray, accelerationStructure, mask, params);

    i.next();
    
    IntersectionResult intersection;
    
    intersection.type = i.get_committed_intersection_type();
    intersection.distance = i.get_committed_distance();
    intersection.primitive_id = i.get_committed_primitive_id();
    intersection.geometry_id = i.get_committed_geometry_id();
    intersection.triangle_barycentric_coord = i.get_committed_triangle_barycentric_coord();
    intersection.instance_id = i.get_committed_instance_id();
    intersection.object_to_world_transform = i.get_committed_object_to_world_transform();
    
    return intersection;
}

void sampleBXDF(thread SampledMaterial& material) {
    if (material.metallic > 0.5f) {
        material.BXDFs = CONDUCTOR;
    } else if (material.refraction > 1.01f) {
        material.BXDFs = SPECULAR_TRANSMISSION;
    } else {
        material.BXDFs = DIFFUSE;
    }
}

SampledMaterial sampleMaterial(Material material, float2 uv, constant Textures* textures) {
    SampledMaterial sampledMaterial;
    constexpr sampler textureSampler(address::repeat, filter::linear);
    bool transparent = false;
    
    if (material.color.textureIndex >= 0) {
        float4 texColor = textures->textures[material.color.textureIndex].sample(textureSampler, uv);
        transparent = texColor.a < 0.8f; // cutoff seems to work
        sampledMaterial.color = material.color.value * (transparent ? 1.0f : texColor.rgb);
    } else {
        sampledMaterial.color = material.color.value;
    }
        
    if (material.roughness.textureIndex >= 0) {
        float4 texValue = textures->textures[material.roughness.textureIndex].sample(textureSampler, uv);
        sampledMaterial.roughness = texValue.g;
    } else {
        sampledMaterial.roughness = material.roughness.value;
    }
    
    if (material.metallic.textureIndex >= 0) {
        float4 texValue = textures->textures[material.metallic.textureIndex].sample(textureSampler, uv);
        sampledMaterial.metallic = texValue.b;
    } else {
        sampledMaterial.metallic = material.metallic.value;
    }
    
    sampledMaterial.refraction = material.refraction.value;
    sampleBXDF(sampledMaterial);
    
    if (transparent) {
        sampledMaterial.BXDFs = SPECULAR_TRANSMISSION;
        sampledMaterial.roughness = 0.0f;
        sampledMaterial.color = 1.0f - sampledMaterial.color;
        sampledMaterial.refraction = 1.0f;
    }
    
    return sampledMaterial;
}

SurfaceInteraction getSurfaceInteraction(ray ray,
                                         IntersectionResult intersection,
                                         device void *resources,
                                         device MTLAccelerationStructureInstanceDescriptor *instances,
                                         instance_acceleration_structure accelerationStructure,
                                         constant int* instanceLightIndices,
                                         int resourcesStride,
                                         constant Textures* textures
                                         )
{
    SurfaceInteraction surfaceInteraction;
    surfaceInteraction.position = ray.origin + ray.direction * intersection.distance;
    
    unsigned int instanceIndex = intersection.instance_id;
    unsigned int mask = instances[instanceIndex].mask;
    float4x3 objectToWorldTransform = intersection.object_to_world_transform;

    unsigned primitiveIndex = intersection.primitive_id;
    unsigned int resourceIndex = instances[instanceIndex].accelerationStructureIndex;
    float2 barycentric_coords = intersection.triangle_barycentric_coord;
    
    device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
    float3 objectNormal = interpolateVertexAttribute(triangleResources.vertexNormals, primitiveIndex, barycentric_coords);
    float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldTransform));
    surfaceInteraction.normal = worldNormal;
    
    float2 uv = interpolateVertexAttribute(triangleResources.vertexUVs, primitiveIndex, barycentric_coords);

    Material material = triangleResources.vertexMaterials[primitiveIndex];
    
    int primitiveLightIndex = triangleResources.vertexPrimitiveLightIndices[primitiveIndex];
    if (primitiveLightIndex != -1) {
        surfaceInteraction.lightIndex = instanceLightIndices[instanceIndex] + primitiveLightIndex;
        constexpr sampler textureSampler(address::repeat, filter::linear);
        
        surfaceInteraction.emission = material.emission.value;
        if (material.emission.textureIndex != -1) {
            surfaceInteraction.emission *= textures->textures[material.emission.textureIndex].sample(textureSampler, uv).rgb;
        }
    } else {
        surfaceInteraction.lightIndex = -1;
        surfaceInteraction.emission = float3(0.0f);
    }
        
    surfaceInteraction.material = sampleMaterial(material, uv, textures);

    return surfaceInteraction;
}

bool isVisible(float3 pos1, float3 normal1,
               float3 pos2, float3 normal2,
               device void *resources,
               device MTLAccelerationStructureInstanceDescriptor *instances,
               instance_acceleration_structure accelerationStructure
               )
{
    float3 w = pos2 - pos1;
    float dist = length(w);
    w /= dist;
    
    float epsilon1 = calculateEpsilon(pos1);
    float epsilon2 = calculateEpsilon(pos2);
    
    float3 offsetOrigin = pos1 + calculateOffset(normal1, w, epsilon1);
    float3 offsetTarget = pos2 + calculateOffset(normal2, -w, epsilon2);

    float3 offsetDir = offsetTarget - offsetOrigin;
    float offsetDist = length(offsetDir);
    offsetDir /= offsetDist;
    
    ray shadowRay;
    shadowRay.origin = offsetOrigin;
    shadowRay.direction = offsetDir;
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = offsetDist * (1.0f - 1e-3f);
    
    IntersectionResult intersection = intersect(shadowRay,
                                                RAY_MASK_SHADOW,
                                                resources,
                                                instances,
                                                accelerationStructure,
                                                true);
    
    return intersection.type == intersection_type::none;
}
