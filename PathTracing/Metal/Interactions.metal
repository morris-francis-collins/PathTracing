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

struct TriangleResources
{
    device float3 *vertexNormals;
    device float3 *vertexColors;
    device Material *vertexMaterials;
    device float2 *vertexUVs;
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

SurfaceInteraction getSurfaceInteraction(ray ray,
                                         IntersectionResult intersection,
                                         device void *resources,
                                         device MTLAccelerationStructureInstanceDescriptor *instances,
                                         instance_acceleration_structure accelerationStructure,
                                         device int *lightIndices,
                                         int resourcesStride,
                                         array<texture2d<float>, MAX_TEXTURES> textureArray
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
    
    Material material = triangleResources.vertexMaterials[primitiveIndex];
    surfaceInteraction.material = material;
    
    surfaceInteraction.hitLight = mask & GEOMETRY_MASK_LIGHT;
    surfaceInteraction.lightIndex = lightIndices[instanceIndex];
    
    float2 uv = interpolateVertexAttribute(triangleResources.vertexUVs, primitiveIndex, barycentric_coords);
    uv.y = 1 - uv.y;
    
    constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);

    if (material.textureIndex != -1) {
        texture2d<float> texture = textureArray[material.textureIndex];
        float4 textureValue = texture.sample(textureSampler, uv);
        float3 textureColor = textureValue.w > 0.0f ? textureValue.xyz : 1.0f;
        surfaceInteraction.textureColor = textureColor;
    } else {
        surfaceInteraction.textureColor = float3(1.0f);
    }

    return surfaceInteraction;
}
