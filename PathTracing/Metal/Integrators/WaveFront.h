//
//  WaveFront.h
//  PathTracing
//
//  Created on 2/7/26.
//

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"
#include "Samplers.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

enum class BucketFlags : uint {
    Emission = 1u << 0,
    Specular = 1u << 1,
    Diffuse = 0u << 2,
    Conductor = 1u << 2,
    DielectricReflection = 2u << 2,
    DielectricTransmission = 3u << 2,
    
    LobeMask = 3u << 2,
};

inline BucketFlags operator|(BucketFlags lhs, BucketFlags rhs) {
    return static_cast<BucketFlags>(static_cast<uint>(lhs) | static_cast<uint>(rhs));
}

inline BucketFlags operator&(BucketFlags lhs, BucketFlags rhs) {
    return static_cast<BucketFlags>(static_cast<uint>(lhs) & static_cast<uint>(rhs));
}

inline thread BucketFlags& operator|=(thread BucketFlags& lhs, BucketFlags rhs) {
    lhs = lhs | rhs;
    return lhs;
}

inline thread BucketFlags& operator&=(thread BucketFlags& lhs, BucketFlags rhs) {
    lhs = lhs & rhs;
    return lhs;
}

BucketFlags bucketMaterial(SampledMaterial material) {
    BucketFlags flags = static_cast<BucketFlags>(0u);
    
    if (!isBlack(dot(material.emission, material.emission))) {
        flags |= BucketFlags::Emission;
    }
    
    if (material.roughness < 0.01f) {
        flags |= BucketFlags::Specular;
    }
    
    if (material.transmission > 0.0f) {
        flags |= BucketFlags::DielectricTransmission;
    } else if (material.metallic > 0.5f) {
        flags |= BucketFlags::Conductor;
    } else if (material.ior >= 1.5f) {
        flags |= BucketFlags::DielectricTransmission;
    } else {
        flags |= BucketFlags::Diffuse;
    }
    
    return flags;
}

#endif
