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

void debug(float x) {
    os_log_default.log_info("%f", x);
}

void debug(float3 w) {
    os_log_default.log_info("mag : %f : float3(%f, %f, %f)", length(w), w.x, w.y, w.z);
}
