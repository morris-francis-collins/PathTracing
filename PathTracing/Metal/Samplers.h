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

inline uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

inline uint owenScramble(uint x, uint seed) {
    x = reverse_bits(x);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    x = reverse_bits(x);
    return x;
}

//struct Sampler {
//    uint state;
//    
//    Sampler(uint pixelIndex, uint frameIndex) {
//        seed(pixelIndex, frameIndex);
//    }
//    
//    void seed(uint pixelIndex, uint frameIndex) {
//        state = pixelIndex + frameIndex * 719393u;
//        r(); r();
//    }
//    
//    float r() {
//        state = state * 747796405u + 2891336453u;
//        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
//        word = (word >> 22u) ^ word;
//        return float(word) / 4294967295.0f;
//    }
//
//    float2 r2() {
//        return float2(r(), r());
//    }
//    
//    float3 r3() {
//        return float3(r(), r(), r());
//    }
//}

struct Sampler {
    uint sampleIndex;
    uint pixelSeed;
    uint dimension;
    constant uint* directionVectors;

    Sampler(uint pixelIndex, uint frameIndex, constant uint* dv) : sampleIndex(frameIndex), pixelSeed(hash(pixelIndex)), dimension(0u), directionVectors(dv) {}
    
    uint sobolValue(uint dim) {
        uint result = 0;
        uint n = sampleIndex;
        
        while (n != 0) {
            result ^= directionVectors[dim * 32 + ctz(n)];
            n &= n - 1;
        }
        
        return result;
    }

    float r() {
        uint raw = sobolValue(dimension);
        uint seed = hash(pixelSeed ^ (dimension * 0x9e3779b9u));
        uint scrambled = owenScramble(raw, seed);
        dimension++;
        
        return (float(scrambled) + 0.5f) / 4294967296.0f;
    }

    float2 r2() {
        float u = r();
        float v = r();
        return float2(u, v);
    }
    
    float3 r3() {
        float u = r();
        float v = r();
        float w = r();
        return float3(u, v, w);
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
