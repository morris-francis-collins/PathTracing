//
//  Shaders.metal
//  PathTracing
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "Shaders.h"

using namespace metal;
using namespace raytracing;
 
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

fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> renderTexture,
                             texture2d<float> referenceTexture,
                             constant unsigned int& displayMode)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    if (displayMode == 0) { // render
        return float4(reinhardTonemap(renderTexture.sample(sam, in.uv).rgb), 1.0f);
    } else if (displayMode == 1) { // reference
        return float4(reinhardTonemap(referenceTexture.sample(sam, in.uv).rgb), 1.0f);
    } else if (displayMode == 2) { // false color
        float3 render = renderTexture.sample(sam, in.uv).rgb;
        float3 reference = referenceTexture.sample(sam, in.uv).rgb;
        float3 diff = abs(render - reference);
        float3 denom = max(reference, float3(0.01f));

        float err = (diff.r / denom.r + diff.g / denom.g + diff.b / denom.b) / 3.0f;
        float t = saturate(log2(max(err, 1e-6f)) / 4.0f + 1.0f);
        float3 falseColor = mix(float3(0, 1.0, 0), float3(1, 0, 0), t);
        return float4(falseColor, 1.0f);
    }
    
    return float4(1.0f, 0.0f, 1.0f, 1.0f); // never should happen
}
