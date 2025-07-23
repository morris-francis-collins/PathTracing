//
//  Utility.metal
//  PathTracing
//
//  Created on 7/18/25.
//

#include <metal_stdlib>
#include "Utility.h"

using namespace metal;
using namespace raytracing;

//HaltonSampler::HaltonSampler(uint _index, uint _frameIndex)
//  : index(_index), dimension(0), frameIndex(_frameIndex) {}
//        
//float HaltonSampler::halton(uint i, uint d) const {
//        uint b = primes[d];
//
//        float f = 1.0f;
//        float invB = 1.0f / b;
//
//        float r = 0;
//
//        while (i > 0)
//        {
//            f = f * invB;
//            r = r + f * (i % b);
//            i = i / b;
//        }
//        
//        return r;
//}
//
//uint HaltonSampler::hash(uint input) const {
//        uint state = input * 747796405u + 2891336453u;
//        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
//        return (word >> 22u) ^ word;
//    }
//
//float HaltonSampler::scrambledHalton(uint index, uint dimension, uint frameIndex) const {
//        return fract(halton(index, dimension) + float(hash(dimension + frameIndex * 719)) / 4294967296.0f);
//}
//    
//float HaltonSampler::r(){
//    return scrambledHalton(index, dimension++, frameIndex);
//}

void debug(float x) {
    os_log_default.log_info("%f", x);
}

void debug(float3 w) {
    os_log_default.log_info("mag : %f : float3(%f, %f, %f)", length(w), w.x, w.y, w.z);
}
