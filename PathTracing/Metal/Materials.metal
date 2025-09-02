//
//  Materials.metal
//  PathTracing
//
//  Created on 7/19/25.
//

#include <metal_stdlib>
#include <simd/simd.h>
#include "Materials.h"
#include "Interactions.h"

// MARK: Diffuse

BSDFSample sampleDiffuseBRDF(float3 wi, float3 n, Material material, float2 r2) {
    float3 woLocal = sampleCosineWeightedHemisphere(r2);
    float3 wo = alignHemisphereWithNormal(woLocal, n);
    
    float3 BSDF = material.color / M_PI_F;
    float PDF = max(dot(wo, n), 0.0f) / M_PI_F;
    
    return BSDFSample(BSDF, wo, PDF);
}

float3 diffuseBRDF(float3 wi, float3 wo, Material material) {
    return material.color / M_PI_F;
}

float diffusePDF(float3 wi, float3 wo, float3 n) {
    return max(dot(wo, n), 0.0f) / M_PI_F;
}

// MARK: Conductor

BSDFSample sampleConductorBRDF(float3 wi, float3 n, Material material, float2 r2) {
    float cosIN = dot(wi, n);
    
    if (material.roughness < 0.01f) {
        float3 wo = reflect(-wi, n); // wi pointing into surface
        float3 fresnel = conductorFresnel(cosIN, material);
        float3 BSDF = fresnel / dot(wo, n);
        return BSDFSample(BSDF, wo, 1.0f, true);
    }
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
    
    float alpha = material.roughness * material.roughness;
    
    float3 localH = sampleGGXNormal(wiLocal, alpha, alpha, r2);
    float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
    float3 woLocal = reflect(-wiLocal, localH);
    float3 wo = reflect(-wi, H);
    
    if (wiLocal.z * woLocal.z <= 0.0f) {
        return BSDFSample(float3(0.0f), float3(1.0f), 1.0f);
    }
    
    float cosON = dot(wo, n);
    float cosIH = dot(wi, H);
    
    float D = D_GGX(localH, alpha, alpha);
    float G1 = G1_Smith(wiLocal, alpha);
    
    float PDF = (D * G1) / (4.0f * cosIN);
    
    float G = G_Smith(wiLocal, woLocal, alpha);
    float3 fresnel = conductorFresnel(cosIH, material);

    float3 BSDF = (D * G * fresnel) / (4.0f * cosIN * cosON);
    
    return BSDFSample(BSDF, wo, PDF);
}

float3 conductorBSDF(float3 wi, float3 wo, float3 n, Material material) {
    if (material.roughness < 0.01f)
        return float3(0.0f);
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float3 wm = normalize(wiLocal + woLocal);
    
    if (wiLocal.z * woLocal.z < 0.0f)
        return float3(0.0f);

    float cosIN = dot(wi, n);
    float cosON = dot(wo, n);
    float cosIH = dot(wiLocal, wm);
                
    float alpha = material.roughness * material.roughness;
    float D = D_GGX(wm, alpha, alpha);
    float G = G_Smith(wiLocal, woLocal, alpha);
    float3 F = conductorFresnel(cosIH, material);
    
    return (D * G * F) / (4.0f * cosIN * cosON);
}

float conductorPDF(float3 wi, float3 wo, float3 n, Material material) {
    if (material.roughness < 0.01f)
        return 0.0f;
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float3 wm = normalize(wiLocal + woLocal);
    
    if (wiLocal.z * woLocal.z < 0.0f)
        return 0.0f;
    
    float alpha = material.roughness * material.roughness;
    float D = D_GGX(wm, alpha, alpha);
    float G1 = G1_Smith(wiLocal, alpha);
    float cosIN = dot(wi, n);

    return (D * G1) / (4.0f * cosIN);
}

// MARK: Dielectric

BSDFSample sampleDielectricBSDF(float3 wi, float3 n, Material material, float r, float2 r2) {
    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    float eta = entering ? 1.0f / material.refraction : material.refraction;
    cosIN = abs(cosIN);
    
    if (material.roughness < 0.01f) {
        float fresnel_R = dielectricFresnel(cosIN, eta);
        float fresnel_T = 1.0f - fresnel_R;
        
        float reflectChance = fresnel_R;
        
        if (r < reflectChance) {
            float3 wo = reflect(-wi, n);
            float3 cosON = abs(dot(wo, n));
            float3 BSDF = fresnel_R / cosON;
            
            return BSDFSample(BSDF, wo, reflectChance, true);
        } else {
            float3 wo = refract(-wi, n, eta);
            
            if (length_squared(wo) < 1e-5f) // should never happen
                return BSDFSample(float3(0.0f), float3(0.0f), 0.0f);
            
            float3 cosON = abs(dot(wo, n));
            float3 BSDF = fresnel_T / cosON;

            return BSDFSample(BSDF, wo, 1.0f - reflectChance, true, true);
        }
    }
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
    
    float alpha = material.roughness * material.roughness;
    
    float3 localH = sampleGGXNormal(wiLocal, alpha, alpha, r2);
    float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
    
    float cosIH = dot(wi, H);
    float fresnel_R = dielectricFresnel(cosIH, eta);
    float fresnel_T = 1.0f - fresnel_R;
    
    float reflectChance = fresnel_R;
    
    if (r < reflectChance) {
        float3 wo = reflect(-wi, H);
        float3 woLocal = reflect(-wiLocal, localH);

        float cosON = dot(wo, n);
        
        float D = D_GGX(localH, alpha, alpha);
        float G1 = G1_Smith(wiLocal, alpha);
    
        float PDF = (D * G1) / (4.0f * cosIN);
        
        float G = G_Smith(wiLocal, woLocal, alpha);

        float3 BSDF = (D * G * fresnel_R) / (4.0f * cosIN * cosON);
        
        return BSDFSample(BSDF, wo, PDF * reflectChance, false, false);
        
    } else {
        float3 wo = refract(-wi, H, eta);
        float3 woLocal = refract(-wiLocal, localH, eta);
        float cosON = dot(wo, n);
        float cosOH = dot(wo, H);
                
        float D = D_GGX(localH, alpha, alpha);
        float G1 = G1_Smith(wiLocal, alpha);

        float denomF = eta * cosIH + cosOH;
        float jacobianF = (eta * eta * abs(cosOH)) / (denomF * denomF);
        float PDF = (D * G1 * abs(cosIH) / abs(cosIN)) * jacobianF;
        
        float G = G_Smith(wiLocal, woLocal, alpha);
        
        float denomBSDF = eta * cosIH + cosOH;
        float jacobian = eta * eta * abs(cosIH * cosOH) / (denomBSDF * denomBSDF);
        float3 BSDF = D * G * fresnel_T * jacobian / abs(cosIN * cosON);
        
        return BSDFSample(BSDF, wo, PDF * (1.0f - reflectChance), false, true);
    }
}

float3 dielectricBSDF(float3 wi, float3 wo, float3 n, Material material) {
    if (material.roughness < 0.01f) {
        return float3(0.0f);
    }
    
    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    float eta = entering ? 1.0f / material.refraction : material.refraction;
    cosIN = abs(cosIN);
    float cosON = dot(wo, n);
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
        
    float alpha = material.roughness * material.roughness;
    float fresnel_R = dielectricFresnel(cosIN, eta);
    float fresnel_T = 1.0f - fresnel_R;
    
    if (cosIN * cosON > 0.0f) {
        float3 wm = normalize(wiLocal + woLocal);
        return D_GGX(wm, alpha, alpha) * G_Smith(wiLocal, woLocal, alpha) * fresnel_R / abs(4.0f * cosIN * cosON);
    } else {
        float3 wm = normalize(wiLocal * eta + woLocal);
        float cosIH = dot(wiLocal, wm);
        float cosOH = dot(woLocal, wm);
        float denom = eta * cosIH + cosOH;
        float jacobian = eta * eta * abs(cosIH * cosOH) / (denom * denom);
        return D_GGX(wm, alpha, alpha) * G_Smith(wiLocal, woLocal, alpha) * fresnel_T * jacobian / abs(cosIN * cosON);
    }
}

float dielectricPDF(float3 wi, float3 wo, float3 n, Material material) {
    if (material.roughness < 0.01f) {
        return 0.0f;
    }
    
    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    float eta = entering ? 1.0f / material.refraction : material.refraction;
    cosIN = abs(cosIN);
    float cosON = dot(wo, n);
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    
    float alpha = material.roughness * material.roughness;
    float fresnel_R = dielectricFresnel(cosIN, eta);
    float fresnel_T = 1.0f - fresnel_R;

    float reflectChance = fresnel_R / (fresnel_R + fresnel_T);
    
    if (cosIN * cosON > 0.0f) {
        float3 wm = normalize(wiLocal + woLocal);
        float PDF = D_GGX(wm, alpha, alpha) * G1_Smith(wiLocal, alpha) / (4.0f * cosIN);
        return PDF * reflectChance;
    } else {
        float3 wm = normalize(wiLocal * eta + woLocal);
        float cosIH = dot(wiLocal, wm);
        float cosOH = dot(woLocal, wm);

        float D = D_GGX(wm, alpha, alpha);
        float G1 = G1_Smith(wiLocal, alpha);

        float denomF = eta * cosIH + cosOH;
        float jacobianF = (eta * eta * abs(cosOH)) / (denomF * denomF);
        float PDF = (D * G1 * abs(cosIH) / abs(cosIN)) * jacobianF;
        
        return PDF * (1.0f - reflectChance);
    }
}

// MARK: Other

BSDFSample sampleBXDF(float3 wi, float3 n, Material material, float3 r3) {
    BSDFSample bsdfSample;
    
    if (material.BXDFs == DIFFUSE) {
        bsdfSample = sampleDiffuseBRDF(wi, n, material, r3.xy);
    } else if (material.BXDFs == CONDUCTOR) {
        bsdfSample = sampleConductorBRDF(wi, n, material, r3.xy);
    } else if (material.BXDFs == SPECULAR_TRANSMISSION) {
        bsdfSample = sampleDielectricBSDF(wi, n, material, r3.x, r3.yz);
    } else {
        bsdfSample = BSDFSample(float3(0.0f), float3(0.0f), 1.0f);
//        DEBUG("sampleBXDF - BXDF not found.");
    }
    
    bsdfSample.BSDF = max(bsdfSample.BSDF, float3(0.0f));
    return bsdfSample;
}

float3 getBXDF(float3 wi, float3 wo, float3 n, Material material) {
    float3 BXDF;
    
    if (material.BXDFs == DIFFUSE) {
        BXDF = diffuseBRDF(wi, wo, material);
    } else if (material.BXDFs == CONDUCTOR) {
        BXDF = conductorBSDF(wi, wo, n, material);
    } else if (material.BXDFs == SPECULAR_TRANSMISSION) {
        BXDF = dielectricBSDF(wi, wo, n, material);
    } else {
        BXDF = float3(0.0f);
//        DEBUG("getBXDF - BXDF not found. BXDF: %d", material.BXDFs);
    }

    return max(BXDF, float3(0.0f));
}

float getPDF(float3 wi, float3 wo, float3 n, Material material) {
    float PDF;
    
    if (material.BXDFs == DIFFUSE) {
        PDF = diffusePDF(wi, wo, n);
    } else if (material.BXDFs == CONDUCTOR) {
        PDF = conductorPDF(wi, wo, n, material);
    } else if (material.BXDFs == SPECULAR_TRANSMISSION) {
        PDF = dielectricPDF(wi, wo, n, material);
    } else {
        PDF = 0.0f;
//        DEBUG("getPDF - PDF not found.");
    }

    return max(PDF, 0.0f);
}
