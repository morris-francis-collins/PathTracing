//
//  Samplers.h
//  PathTracing
//
//  Created on 9/9/25.
//

#pragma once
#include <simd/simd.h>

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

enum SamplerType : unsigned int {
    HALTON = 0,
    SOBOL = 1,
};

struct Sampler {
    uint state;
    
    Sampler(uint pixelIndex, uint frameIndex) {
        seed(pixelIndex, frameIndex);
    }
    
    void seed(uint pixelIndex, uint frameIndex) {
        state = pixelIndex + frameIndex * 719393u;
        r(); r();
    }
    
    float r() {
        state = state * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        word = (word >> 22u) ^ word;
        return float(word) / 4294967295.0f;
    }
    
    float2 r2() {
        return float2(r(), r());
    }
    
    float3 r3() {
        return float3(r(), r(), r());
    }
};

inline uint init_prng(uint pixelIndex, uint frameIndex) {
    return pixelIndex + frameIndex * 719393u;
}

inline float prng(device uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) / 4294967295.0f;
}

inline float prng(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) / 4294967295.0f;
}


#endif
