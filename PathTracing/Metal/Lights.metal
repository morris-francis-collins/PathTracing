//
//  Lights.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include "Lights.h"
#include "Interactions.h"

using namespace metal;
using namespace raytracing;

float getLightPower(device const Light& light) {
    switch (light.type) {
        case POINT_LIGHT:
            return 4.0f * M_PI_F * calculateLuminance(light.color);
        case AREA_LIGHT:
            return calculateLuminance(light.color) * light.totalArea;
        case DIRECTIONAL_LIGHT:
            return M_PI_F * SCENE_RADIUS * SCENE_RADIUS * calculateLuminance(light.color);
        case ENVIRONMENT_MAP:
            return 4.0f * M_PI_F * M_PI_F * SCENE_RADIUS * SCENE_RADIUS * ENVIRONMENT_MAP_SCALE * calculateLuminance(light.color);
        default:
            return 0.0f;
    }
}

float getLightSelectionPDF(device Light *lights, constant Uniforms& uniforms, unsigned int lightIndex) {
    float strategy, sum = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        if (i == lightIndex) {
            strategy = getLightPower(lights[i]);
        }
        sum += getLightPower(lights[i]);
    }
    
    return strategy / sum;
}

float environmentLightSamplePDF(float2 uv, device float *environmentMapCDF) {
    int x = clamp(int(uv.x * float(ENVIRONMENT_MAP_WIDTH)), 0, ENVIRONMENT_MAP_WIDTH - 1);
    int y = clamp(int(uv.y * float(ENVIRONMENT_MAP_HEIGHT)), 0, ENVIRONMENT_MAP_HEIGHT - 1);
    int i = y * ENVIRONMENT_MAP_WIDTH + x;

    float pixelPDF = environmentMapCDF[i] - (i > 0 ? environmentMapCDF[i - 1] : 0.0f);
    return pixelPDF * float(ENVIRONMENT_MAP_WIDTH * ENVIRONMENT_MAP_HEIGHT) / (2.0f * M_PI_F * M_PI_F * sin(uv.y * M_PI_F));
}

float getLightSamplePDF(thread Light& light) {
    switch (light.type) {
        case POINT_LIGHT:
            return 0.0f;
        case AREA_LIGHT:
            return 1.0f / light.totalArea;
        case DIRECTIONAL_LIGHT:
            return 0.0f;
        default:
            return 0.0f;
    }
}

LightSample samplePointLight(float3 position, thread Light& pointLight) {
    float3 wo = pointLight.position - position;
    float d = length(wo);
    wo /= d;
    return LightSample(pointLight.position, float3(0.0f), wo, pointLight.color, d, 1.0f);
}

LightSample sampleAreaLight(float3 position,
                            thread Light& areaLight,
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
    float3 lightPosition = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    float3 normal = normalize(cross(edge1, edge2));
    
    float3 wo = lightPosition - position;
    float d = length(wo);
    wo /= d;

    return LightSample(position, normal, wo, areaLight.color, d, 1.0f / areaLight.totalArea);
}

LightSample sampleDirectionalLight(float3 position, thread Light& directionalLight) {
    float3 wo = -directionalLight.direction;
    float3 lightPosition = position + 2.0f * SCENE_RADIUS * wo;
    return LightSample(lightPosition, float3(0.0f), wo, directionalLight.color, 1.0f, 1.0f);
}

LightSample sampleEnvironmentMap(float3 position,
                                 thread Light& environmentMap,
                                 texture2d<float> environmentMapTexture,
                                 device float *environmentMapCDF,
                                 float r1
                                 )
{
    int l = 0, r = ENVIRONMENT_MAP_WIDTH * ENVIRONMENT_MAP_HEIGHT - 1;
    
    while (l < r) {
        int m = l + (r - l) / 2;
        
        if (environmentMapCDF[m] < r1) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    
    int y = l / ENVIRONMENT_MAP_WIDTH;
    int x = l % ENVIRONMENT_MAP_WIDTH;

    float u = (float(x) + 0.5f) / float(ENVIRONMENT_MAP_WIDTH);
    float v = (float(y) + 0.5f) / float(ENVIRONMENT_MAP_HEIGHT);
    
    constexpr sampler envSampler(min_filter::linear, mag_filter::linear, s_address::repeat, t_address::clamp_to_edge);
    float3 emission = ENVIRONMENT_MAP_SCALE * environmentMapTexture.sample(envSampler, float2(u, v)).xyz;

    float phi = u * 2.0f * M_PI_F;
    float theta = v * M_PI_F;
    
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    
    float3 wo;
    wo.x = -sinTheta * cosPhi;
    wo.y = cosTheta;             // y is up
    wo.z = -sinTheta * sinPhi;
    
    float pixelPDF = environmentMapCDF[l] - (l > 0 ? environmentMapCDF[l - 1] : 0.0f);
    float solidAnglePDF = pixelPDF * float(ENVIRONMENT_MAP_WIDTH * ENVIRONMENT_MAP_HEIGHT) / (2.0f * M_PI_F * M_PI_F * sinTheta);
    float3 lightPosition = position + 2.0f * SCENE_RADIUS * wo;
    
    return LightSample(lightPosition, float3(0.0f), wo, emission, 1.0f, solidAnglePDF);
}

LightSample sampleLight(float3 position,
                        thread Light& light,
                        device LightTriangle *lightTriangles,
                        texture2d<float> environmentMapTexture,
                        device float *environmentMapCDF,
                        float3 r3
                        )
{
    switch (light.type) {
        case POINT_LIGHT:
            return samplePointLight(position, light);
        case AREA_LIGHT:
            return sampleAreaLight(position, light, lightTriangles, r3);
        case DIRECTIONAL_LIGHT:
            return sampleDirectionalLight(position, light);
        case ENVIRONMENT_MAP:
            return sampleEnvironmentMap(position, light, environmentMapTexture, environmentMapCDF, r3.x);
        default:
            return LightSample(float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f), 0.0f, 0.0f);
    }
}

Light selectLight(device Light *lights,
                  device LightTriangle *lightTriangles,
                  constant Uniforms& uniforms,
                  float r,
                  thread float& selectionPDF
                  )
{
    float weights[MAX_LIGHTS];
    float totalWeight = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        float power = getLightPower(lights[i]);
                        
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

    return lights[idx];
}
