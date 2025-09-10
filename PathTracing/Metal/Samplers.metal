//
//  Samplers.metal
//  PathTracing
//
//  Created on 9/9/25.
//

#include <metal_stdlib>
#include "Samplers.h"
#include "Utility.h"

using namespace metal;
using namespace raytracing;

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

uint reverseBits(uint n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

uint owenScramble(uint value, uint scramble) {
    value = reverseBits(value);
    value ^= scramble;
    value = reverseBits(value);
    return value;
}

float haltonSample(uint i, uint d, uint frameIndex) {
    uint b = primes[d];
    
    float f = 1.0f;
    float invB = 1.0f / b;
    
    float r = 0;
    
    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }
    
    return fract(r + float(hash(d + frameIndex * 719)) / 4294967296.0f);
}

float sobolSample(uint i, uint d, uint frameIndex) {
    uint result = 0;

    for (uint bit = 0; bit < 32 && i > 0; bit++) {
        if (i & 1)
            result ^= SOBOL_VALUES[d][bit];
        i >>= 1;
    }

    result = owenScramble(result, hash(frameIndex + d * 0x1337));
    return result * 2.3283064365386963e-10f; // 1/2^32
}

float Sampler::r() {
    switch (type) {
        case HALTON:
            return haltonSample(index, dimension++, frameIndex);
        case SOBOL:
            return sobolSample(index, dimension++, frameIndex);
        default:
            DEBUG("Sampler r() not handled.");
            return 0.0f;
    }
}

float2 Sampler::r2() {
    return float2(r(), r());
}

float3 Sampler::r3() {
    return float3(r(), r(), r());
}
