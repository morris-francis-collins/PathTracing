//
//  Debug.metal
//  PathTracing
//
//  Created on 3/26/26.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "Debug.h"

using namespace metal;
using namespace raytracing;

kernel void debugSurfaceProperties(uint2 tid [[thread_position_in_grid]],
                                   constant int* instanceLightIndices,
                                   constant Textures* textures,
                                   constant Material* materials,
                                   
                                   device MTLAccelerationStructureInstanceDescriptor *instances,
                                   instance_acceleration_structure accelerationStructure,

                                   constant Uniforms& uniforms,
                                   constant DebugType& debugType,
                                   
                                   texture2d<float, access::write> image,
                                   texture2d<float> environmentMapTexture
                             )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    image.write(float4(0.0f), tid);
    ray ray = generateRay(static_cast<float2>(tid), uniforms);
    float3 color = float3(0.0f);

    IntersectionResult intersection = intersect<false>(ray, accelerationStructure);
    
    if (intersection.type == intersection_type::none) {
        float2 uv = getEnvironmentMapUV(ray.direction);
        color = environmentMapEmission(uv, environmentMapTexture);
        image.write(float4(color, 1.0f), tid);
        return;
    }
    
    SurfaceInteraction surfaceInteraction = getSurfaceInteraction(ray, intersection, instances, accelerationStructure, instanceLightIndices, textures, materials);
    SampledMaterial material = surfaceInteraction.material;
    
    switch (debugType) {
        case Color:
            color = material.color; break;
        case Roughness:
            color = scalarToColor(material.roughness); break;
        case Metallic:
            color = scalarToColor(material.metallic); break;
        case IOR:
            color = scalarToColor(saturate((material.ior - 1.0f) / 1.5f)); break;
        case Transmission:
            color = scalarToColor(material.transmission); break;
        case Clearcoat:
            color = scalarToColor(material.clearcoat); break;
        case ClearcoatRoughness:
            color = scalarToColor(material.clearcoatRoughness); break;
        case ThicknessFactor:
            color = scalarToColor(saturate(material.thicknessFactor)); break;
        case AttenuationColor:
            color = material.attenuationColor; break;
        case AttenuationDistance:
            color = scalarToColor(saturate(log(1.0f + material.attenuationDistance) / log(101.0f))); break;
        case Alpha:
            color = scalarToColor(material.alpha); break;
        case AlphaMode:
            if (material.alphaMode == 0)
                color = float3(1.0f, 0.0f, 0.0f);
            else if (material.alphaMode == 1)
                color = float3(0.0f, 1.0f, 0.0f);
            else
                color = float3(0.0f, 0.0f, 1.0f);
            break;
        case Emission:
            color = material.emission / (1.0f + material.emission); break;
        case BXDF:
            if (material.BXDFs == DIFFUSE)
                color = float3(1.0f, 0.0f, 0.0f);
            else if (material.BXDFs == CONDUCTOR)
                color = float3(0.0f, 1.0f, 0.0f);
            else if (material.BXDFs == DIELECTRIC_TRANSMISSION)
                color = float3(0.0f, 0.0f, 1.0f);
            else if (material.BXDFs == DIELECTRIC_REFLECTION)
                color = float3(1.0f, 1.0f, 0.0f);
            else
                color = float3(1.0f);
            break;
        case Normal:
            color = surfaceInteraction.normal * 0.5f + 0.5f; break;
    }
    
    image.write(float4(color, 1.0f), tid);
}
