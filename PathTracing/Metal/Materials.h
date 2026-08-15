//
//  Materials.h
//  PathTracing
//
//  Created on 7/19/25.
//

#pragma once

#include "Utility.h"
#include "Samplers.h"

#define DIFFUSE (1 << 0)
#define CONDUCTOR (1 << 1)
#define DIELECTRIC_TRANSMISSION (1 << 2)
#define DIELECTRIC_REFLECTION (1 << 3)

#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2

struct Material {
    vector_float3 colorValue;
    int colorTextureIndex;
    float roughnessValue;
    int roughnessTextureIndex;
    float metallicValue;
    int metallicTextureIndex;
    
    vector_float3 emissionValue;
    int emissionTextureIndex;
    float emissiveStrength;
    
    float specularFactor;
    int specularTextureIndex;
    vector_float3 specularColor;
    int specularColorTextureIndex;
    
    float ior;
    float transmissionValue;
    int transmissionTextureIndex;
    
    float thicknessFactor;
    int thicknessTextureIndex;
    vector_float3 attenuationColor;
    float attenuationDistance;
    
    float diffuseTransmissionFactor;
    int diffuseTransmissionTextureIndex;
    vector_float3 diffuseTransmissionColor;
    int diffuseTransmissionColorTextureIndex;
    
    int normalTextureIndex;
    float normalScale;
    
    float clearcoatValue;
    int clearcoatTextureIndex;
    float clearcoatRoughnessValue;
    int clearcoatRoughnessTextureIndex;
    int clearcoatNormalTextureIndex;
    
    float anisotropyStrength;
    float anisotropyRotation;
    int anisotropyTextureIndex;
    
    vector_float3 sheenColor;
    int sheenColorTextureIndex;
    float sheenRoughness;
    int sheenRoughnessTextureIndex;
    
    float iridescenceFactor;
    int iridescenceTextureIndex;
    float iridescenceIor;
    float iridescenceThicknessMin;
    float iridescenceThicknessMax;
    int iridescenceThicknessTextureIndex;
    
    float dispersion;
    
    unsigned int alphaMode;
    float alphaCutoff;
    
    unsigned int doubleSided;
};

struct SampledMaterial {
    vector_float3 color;
    float ior;
    float roughness;
    float metallic;
    float transmission;
 
    float clearcoat;
    float clearcoatRoughness;
 
    float thicknessFactor;
    vector_float3 attenuationColor;
    float attenuationDistance;
 
    float alpha;
    unsigned int alphaMode;
    float alphaCutoff;
 
    vector_float3 emission;
 
    int BXDFs;
    
#ifdef __METAL_VERSION__
    bool isPerfectSpecular() {
        return roughness < 0.01f and (transmission > 0.01f or metallic > 0.5f);
    }
#endif
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

enum TransportMode: uint8_t {
    Radiance,
    Importance
};

struct BSDFSample {
    float3 BSDF;
    float3 wo;
    float PDF;
    bool delta;
    bool transmitted;
};

BSDFSample sampleBXDF(float3 wi, float3 n, thread SampledMaterial& material, TransportMode transportMode, float3 r3);
float3 getBXDF(float3 wi, float3 wo, float3 n, thread SampledMaterial& material, TransportMode transportMode);
float getPDF(float3 wi, float3 wo, float3 n, thread SampledMaterial& material);

inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;

    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    

    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    float3 up = normal;
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    float3 forward = cross(right, up);

    return sample.x * right + sample.y * up + sample.z * forward;
}

inline float lambda(float3 w, float alpha) {
    float cosTheta2 = w.z * w.z;
    float sinTheta2 = 1.0f - cosTheta2;
    return 0.5f * (sqrt(1.0f + alpha * alpha * sinTheta2 / cosTheta2) - 1.0f);
}

inline float G_Smith(float3 wi, float3 wo, float alpha) {
    float lambda_wi = lambda(wi, alpha);
    float lambda_wo = lambda(wo, alpha);
    return 1.0f / (1.0f + lambda_wi + lambda_wo);
}

inline float G1_Smith(float3 w, float alpha) {
    return 1.0f / (1.0f + lambda(w, alpha));
}

inline float D_GGX(float3 wm, float alpha_x, float alpha_y) {
    float x = wm.x / alpha_x;
    float y = wm.y / alpha_y;
    float z = wm.z;

    float denom = x * x + y * y + z * z;
    return 1.0f / (M_PI_F * alpha_x * alpha_y * denom * denom);
}

inline void createOrthonormalBasis(float3 n, thread float3& t, thread float3& b) {
    float sign = copysign(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b_val = n.x * n.y * a;
    t = float3(1.0f + sign * n.x * n.x * a, sign * b_val, -sign * n.x);
    b = float3(b_val, sign + n.y * n.y * a, -n.y);
    t = normalize(t); // seems to improve performance despite being redundant
    b = normalize(b);
}

inline float2 concentricSampleDisk(float2 u) {
    float2 u_offset = 2.0f * u - 1.0f;
    if (u_offset.x == 0.0f && u_offset.y == 0.0f) return float2(0.0f, 0.0f);

    float theta, r;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = (M_PI_F / 4.0f) * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (u_offset.x / u_offset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

inline float3 sampleGGXNormal(float3 wi, float alpha_x, float alpha_y, float2 r) {
    float3 V = normalize(float3(wi.x * alpha_x, wi.y * alpha_y, wi.z));
    
    float lensq = V.x * V.x + V.y * V.y;
    float3 T1 = lensq > 0.0f ? float3(-V.y, V.x, 0.0f) * rsqrt(lensq) : float3(0.0f, 0.0f, 1.0f);
    float3 T2 = cross(V, T1);
    
    float2 disk = concentricSampleDisk(r);
    float t1 = disk.x;
    float t2 = disk.y;
    
    float s = 0.5f * (1.0f + V.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    
    float3 N = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * V;
    float3 H = normalize(float3(N.x * alpha_x, N.y * alpha_y, max(0.0f, N.z)));
    
    return H;
}

inline float3 conductorFresnel(float cosI, thread SampledMaterial& material) {
    float3 F0 = material.color;
    return F0 + (1.0f - F0) * pow(1.0f - cosI, 5.0f);
}


inline float dielectricFresnel(float cosI, float eta) {
    float sinI2 = 1.0f - cosI * cosI;
    float sinT2 = eta * eta * sinI2;
    
    if (sinT2 >= 1.0f)
        return 1.0f;
    
    float cosT = sqrt(1.0f - sinT2);
    
    float Rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float Rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    
    return 0.5f * (Rs * Rs + Rp * Rp);
}

#endif
