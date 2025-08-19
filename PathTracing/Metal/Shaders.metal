//
//  Shaders.metal
//  PathTracing
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "Shaders.h"

using namespace metal;
using namespace raytracing;

constant unsigned int resourcesStride  [[function_constant(0)]];
constant bool useIntersectionFunctions [[function_constant(1)]];

// Return type for a bounding box intersection function.
struct BoundingBoxIntersection
{
    bool accept; // Whether to accept or reject the intersection.
    float distance; // Distance from the ray origin to the intersection point.
};

typedef BoundingBoxIntersection IntersectionFunction(float3,
                                                     float3,
                                                     float,
                                                     float,
                                                     unsigned int,
                                                     unsigned int,
                                                     device void *);
 
kernel void raytracingKernel(uint2 tid [[thread_position_in_grid]],
                             uint sampleIndex [[thread_index_in_threadgroup]],
                             constant Uniforms & uniforms,
                             texture2d<unsigned int> randomTex,
                             texture2d<float> prevTex,
                             texture2d<float, access::read_write> dstTex,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             constant Light *lights,
                             instance_acceleration_structure accelerationStructure,
                             visible_function_table<IntersectionFunction> intersectionFunctionTable,
                             device atomic_float* splatBuffer,
                             texture2d<float> prevSplat,
                             texture2d<float, access::read_write> splatTex,
                             texture2d<float, access::write> finalImage,
                             texture2d<float> environmentMapTexture,
                             constant LightTriangle *lightTriangles,
                             array<texture2d<float>, MAX_TEXTURES> textureArray [[texture(8)]],
                             constant int *lightIndices,
                             constant float *environmentMapCDF
                             )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    unsigned int offset = randomTex.read(tid).x;
    
    float2 pixel = (float2)tid;
    float2 r = float2(scrambledHalton(offset, 8, uniforms.frameIndex),
                      scrambledHalton(offset, 9, uniforms.frameIndex));
    pixel += r;
    
    HaltonSampler sampler = HaltonSampler(offset, uniforms.frameIndex);
    
    float3 contribution = pathIntegrator(pixel, uniforms, resourcesStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler);
    
//    float3 contribution = bidirectionalPathIntegrator(pixel, uniforms, resourcesStride, resources, instances, accelerationStructure, lights, lightTriangles, lightIndices, environmentMapTexture, environmentMapCDF, textureArray, sampler);
    
    float3 totalSplat = splatTex.read(tid).xyz;
    
    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(tid).xyz;
        prevColor *= uniforms.frameIndex;
        contribution += prevColor;
        contribution /= (uniforms.frameIndex + 1);
        
        float3 previousSplat = prevSplat.read(tid).xyz;
        previousSplat *= uniforms.frameIndex;
        totalSplat += previousSplat;
        totalSplat /= (uniforms.frameIndex + 1);
    }
    

    dstTex.write(float4(contribution, 1.0f), tid);
    splatTex.write(float4(totalSplat, 1.0f), tid);
    finalImage.write(float4(contribution, 1.0f), tid);
}

kernel void clearAtomicBuffer(device atomic_float* atomicBuffer [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]],
                              texture2d<float, access::write> outputTexture [[texture(0)]],
                              uint2 gridSize [[threads_per_grid]]) {
    
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height())
        return;

    uint width = outputTexture.get_width();
    uint index = gid.y * width + gid.x;
    
    if (index < width * outputTexture.get_height()) {
        uint bufferIndex = index;
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3 + 1], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3 + 2], 0.0f, memory_order_relaxed);
    }
}

kernel void finalizeAtomicBuffer(device atomic_float* atomicBuffer [[buffer(0)]],
                                texture2d<float, access::write> currentSplatTexture [[texture(0)]],
                                 texture2d<float, access::read> previousSplatTexture [[texture(1)]],
                                constant Uniforms &uniforms [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= currentSplatTexture.get_width() || gid.y >= currentSplatTexture.get_height())
        return;
    
    uint width = currentSplatTexture.get_width();
    uint index = (gid.y * width + gid.x) * 3;
    
    float3 currentFrameContribution = float3(
        atomic_load_explicit(&atomicBuffer[index + 0], memory_order_relaxed),
        atomic_load_explicit(&atomicBuffer[index + 1], memory_order_relaxed),
        atomic_load_explicit(&atomicBuffer[index + 2], memory_order_relaxed)
    );
    
    float3 previousAccumulation = previousSplatTexture.read(gid).xyz;
    
    float3 newAccumulation;
    if (uniforms.frameIndex > 0) {
        previousAccumulation *= uniforms.frameIndex;
        newAccumulation = previousAccumulation + currentFrameContribution;
        newAccumulation /= (uniforms.frameIndex + 1);
    } else {
        newAccumulation = currentFrameContribution;
    }
    
    currentSplatTexture.write(float4(newAccumulation, 1.0), gid);
}

// Screen filling quad in normalized device coordinates.
constant float2 quadVertices[] =
{
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut
{
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions.
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]])
{
    float2 position = quadVertices[vid];

    CopyVertexOut out;

    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;

    return out;
}

// Simple fragment shader which copies a texture and applies a simple tonemapping function.
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);

    float3 color = tex.sample(sam, in.uv).xyz;

    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which the screen can display.
    color = color / (1.0f + color);

    return float4(color, 1.0f);
}
