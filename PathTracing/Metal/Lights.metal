//
//  Lights.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include "Lights.h"

using namespace metal;
using namespace raytracing;

LightSample sampleLight(thread AreaLight& areaLight,
                        device LightTriangle *lightTriangles,
                        float3 r3
                      )
{
    int left = areaLight.firstTriangleIndex;
    int right = areaLight.firstTriangleIndex + areaLight.triangleCount - 1;
    float target = r3.x;

    while (left < right) {
        int mid = (left + right) / 2;

        if (target < lightTriangles[mid].cdf) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    LightTriangle triangle = lightTriangles[left];
    if (r3.y + r3.z > 1.0f) {
        r3.y = 1 - r3.y;
        r3.z = 1 - r3.z;
    }

    float u = 1.0f - r3.y - r3.z;
    float v = r3.y;
    float w = r3.z;
    
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 position = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    float epsilon = calculateEpsilon(position);

    LightSample lightSample;
    lightSample.normal = normalize(cross(edge1, edge2));
    lightSample.position = position + calculateOffset(lightSample.normal, lightSample.normal, epsilon);
    lightSample.PDF = 1.0f / areaLight.totalArea;
    lightSample.emission = areaLight.color;
    
    return lightSample;
}

AreaLight selectLight(device AreaLight *areaLights,
                      device LightTriangle *lightTriangles,
                      constant Uniforms& uniforms,
                      float r,
                      thread float& selectionPDF
                          )
{
    float weights[MAX_AREA_LIGHTS];
    float totalWeight = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        AreaLight light = areaLights[i];
        float power = calculateLuminance(light.color) * light.totalArea;
                        
        weights[i] = power;
        totalWeight += weights[i];
    }
    
    unsigned int idx = 0;
    
    if (totalWeight < 1e-4f) {
        idx = r * float(uniforms.lightCount);
        selectionPDF = 1.0f / float(uniforms.lightCount);
    } else {
        float random = r * totalWeight;
        float accumWeight = 0.0f;
        
        for (unsigned int i = 0; i < uniforms.lightCount; i++) {
            accumWeight += weights[i];
            
            if (accumWeight >= random) {
                idx = i;
                selectionPDF = weights[i] / totalWeight;
                break;
            }
        }
    }

    return areaLights[idx];
}


