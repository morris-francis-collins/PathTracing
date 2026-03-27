//
//  Debug.h
//  PathTracing
//
//  Created on 3/26/26.
//

#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"
#include "Samplers.h"

enum DebugType: unsigned int {
    Color = 0,
    Roughness,
    Metallic,
    IOR,
    Transmission,
    Clearcoat,
    ClearcoatRoughness,
    ThicknessFactor,
    AttenuationColor,
    AttenuationDistance,
    Alpha,
    AlphaMode,
    Emission,
    BXDF,
    Normal
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

float3 scalarToColor(float t) {
    t = saturate(t);
    float t2 = t * t;
    float t3 = t2 * t;

    float r = 0.08f + 1.8f * t - 0.9f * t2;
    float g = 0.02f + 0.4f * t + 1.2f * t2 - 0.6f * t3;
    float b = 0.20f + 0.8f * t - 2.2f * t2 + 1.4f * t3;
    
    float w = pow(t, 6.0f);
    float3 color = float3(r, g, b) * (1.0f - w) + w;

    return clamp(color, 0.0f, 1.0f);
}

#endif
