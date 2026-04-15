//
//  SPPM.h
//  PathTracing
//
//  Created on 4/12/26.
//

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"
#include "Samplers.h"

#define HASH_TABLE_SIZE 3999971u
#define PHOTON_COUNT (4 * 65536u)
#define ALPHA 0.55f

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

uint hashCell(int3 cell) {
    uint h = (uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u);
    return h % HASH_TABLE_SIZE;
}

uint hashLocation(float3 position, float hashGridSize) {
    int3 cell = int3(floor(position / hashGridSize));
    return hashCell(cell);
}

float epanechnikov(float d2, float r2) {
    if (d2 >= r2) return 0.0f;
    return (2.0f / M_PI_F) * (1.0f / r2) * (1.0f - d2 / r2);
}

#endif
