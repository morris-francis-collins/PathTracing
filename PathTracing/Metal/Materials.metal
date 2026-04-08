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

BSDFSample sampleDiffuseBRDF(float3 wi, float3 n, SampledMaterial material, float2 r2) {
    float3 woLocal = sampleCosineWeightedHemisphere(r2);
    float3 wo = alignHemisphereWithNormal(woLocal, n);
    
    float3 BSDF = material.color / M_PI_F;
    float PDF = max(dot(wo, n), 0.0f) / M_PI_F;
    
    return BSDFSample{BSDF, wo, PDF};
}

float3 diffuseBRDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    float cosIN = dot(wi, n);
    float cosON = dot(wo, n);
    
    if (cosIN <= 0.0f || cosON <= 0.0f)
        return float3(0.0f);
    
    return material.color / M_PI_F;
}

float diffusePDF(float3 wi, float3 wo, float3 n) {
    float cosIN = dot(wi, n);
    float cosON = dot(wo, n);
    
    if (cosIN <= 0.0f || cosON <= 0.0f)
        return 0.0f;

    return max(dot(wo, n), 0.0f) / M_PI_F;
}

// MARK: Conductor

BSDFSample sampleConductorBRDF(float3 wi, float3 n, SampledMaterial material, float2 r2) {
    float cosIN = dot(wi, n);
    
    if (material.roughness < 0.01f) {
        float3 wo = reflect(-wi, n); // wi needs to point into surface for reflect
        float3 fresnel = conductorFresnel(cosIN, material);
        float3 BSDF = fresnel / dot(wo, n);
        return BSDFSample{BSDF, wo, 1.0f, true};
    }
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
    
    float alpha = material.roughness * material.roughness;
    
    float3 localH = sampleGGXNormal(wiLocal, alpha, alpha, r2);
    float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
    float3 woLocal = reflect(-wiLocal, localH);
    float3 wo = reflect(-wi, H);
    
    if (wiLocal.z * woLocal.z <= 0.0f)
        return BSDFSample{float3(0.0f), float3(1.0f), 1.0f};
    
    float cosON = dot(wo, n);
    float cosIH = dot(wi, H);
    
    float D = D_GGX(localH, alpha, alpha);
    float G1 = G1_Smith(wiLocal, alpha);
    
    float PDF = (D * G1) / (4.0f * cosIN);
    
    float G = G_Smith(wiLocal, woLocal, alpha);
    float3 fresnel = conductorFresnel(cosIH, material);

    float3 BSDF = (D * G * fresnel) / (4.0f * cosIN * cosON);
    
    return BSDFSample{BSDF, wo, PDF};
}

float3 conductorBSDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    if (material.roughness < 0.01f)
        return float3(0.0f);
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float3 wm = normalize(wiLocal + woLocal);
    
    if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
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

float conductorPDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    if (material.roughness < 0.01f)
        return 0.0f;
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float3 wm = normalize(wiLocal + woLocal);
    
    if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f)
        return 0.0f;
    
    float alpha = material.roughness * material.roughness;
    float D = D_GGX(wm, alpha, alpha);
    float G1 = G1_Smith(wiLocal, alpha);
    float cosIN = dot(wi, n);

    return (D * G1) / (4.0f * cosIN);
}

// MARK: Dielectric

BSDFSample sampleDielectricBSDF(float3 wi, float3 n, SampledMaterial material, float r, float2 r2) {
    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    float eta = entering ? 1.0f / material.ior : material.ior;
    cosIN = abs(cosIN);
    
    bool allowTransmission = material.BXDFs == DIELECTRIC_TRANSMISSION;
    
    if (material.roughness < 0.01f) {
        float fresnel_R = dielectricFresnel(cosIN, eta);
        float fresnel_T = 1.0f - fresnel_R;
        
        float pr = fresnel_R;
        float pt = allowTransmission ? fresnel_T : 0.0f;
        float pd = allowTransmission ? 0.0f : fresnel_T;
        float sum = pr + pt + pd;
        
        if (r < pr / sum)  {
            float3 wo = reflect(-wi, n);
            float cosON = abs(dot(wo, n));
            float3 BSDF = fresnel_R / cosON;
            return BSDFSample{BSDF, wo, pr / sum, true, false};
            
        } else {
            if (allowTransmission) {
                float3 wo = refract(-wi, n, eta);
                if (length_squared(wo) < 1e-5f) // should never happen
                    return BSDFSample{float3(0.0f), float3(0.0f), 0.0f, false, false};
                float cosON = abs(dot(wo, n));
                float3 BSDF = fresnel_T / cosON;
                return BSDFSample{BSDF, wo, pt / sum, true, true};
                
            } else {
                BSDFSample diffuseSample = sampleDiffuseBRDF(wi, n, material, r2);
                diffuseSample.BSDF *= fresnel_T;
                diffuseSample.PDF *= pd / sum;
                return diffuseSample;
            }
        }
    }
    
    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
    float alpha = material.roughness * material.roughness;
    
    float F0 = (material.ior - 1.0f) / (material.ior + 1.0f);
    F0 *= F0;
    float avgR = F0 + (1.0f - F0) * 0.15f;
    
    float pr = allowTransmission ? 0.5f : avgR; // need to use rough importance sampling here since we dont know GGX normal yet
    float pt = allowTransmission ? 0.5f : 0.0f;
    float pd = allowTransmission ? 0.0f : 1.0f - avgR;
    
    if (r < pr) {
        float3 localH = sampleGGXNormal(wiLocal, alpha, alpha, r2);
        float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
        
        float3 wo = reflect(-wi, H);
        if (dot(wo, n) <= 0.0f)
            return BSDFSample{float3(0.0f), float3(0.0f), 1.0f};
        
        float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
        float cosON = abs(dot(wo, n));
        float cosIH = abs(dot(wi, H));
        
        float fresnel_R = dielectricFresnel(cosIH, eta);
        float D = D_GGX(localH, alpha, alpha);
        float G = G_Smith(wiLocal, woLocal, alpha);
        float G1 = G1_Smith(wiLocal, alpha);
        
        float3 specBSDF = float3(fresnel_R) * D * G / (4.0f * cosIN * cosON);
        float specPDF = (D * G1) / (4.0f * cosIN);
        float PDF = pr * specPDF;
        
        if (!allowTransmission) {
            float3 diffBSDF = (1.0f - fresnel_R) * diffuseBRDF(wi, wo, n, material);
            float diffPDF = diffusePDF(wi, wo, n);
            PDF += pd * diffPDF;
            return BSDFSample{specBSDF + diffBSDF, wo, PDF};
        }
        
        return BSDFSample{specBSDF, wo, PDF};
        
    } else {
        if (allowTransmission) {
            float3 localH = sampleGGXNormal(wiLocal, alpha, alpha, r2);
            float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
            
            float cosIH = dot(wi, H);
            float fresnel_T = 1.0f - dielectricFresnel(abs(cosIH), eta);
            
            float3 wo = refract(-wi, H, eta);
            if (length_squared(wo) < 1e-5f)
                return BSDFSample{float3(0.0f), float3(0.0f), 1.0f};
            
            float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
            float cosON = abs(dot(wo, n));
            float cosOH = dot(wo, H);
            
            float D = D_GGX(localH, alpha, alpha);
            float G1 = G1_Smith(wiLocal, alpha);
            float G = G_Smith(wiLocal, woLocal, alpha);
            
            float denom = eta * cosIH + cosOH;
            float dwm_dwi = abs(cosOH) / (denom * denom);
            float transPDF = D * G1 * abs(cosIH) / cosIN * dwm_dwi;
            float jacobian = eta * eta * abs(cosIH * cosOH) / (denom * denom);
            float3 BSDF = float3(fresnel_T) * D * G * jacobian / (cosIN * cosON);
            
            return BSDFSample{BSDF, wo, pt * transPDF, false, true};
            
        } else {
            float3 woLocal_h = sampleCosineWeightedHemisphere(r2);
            float3 wo = alignHemisphereWithNormal(woLocal_h, n);
            if (dot(wo, n) <= 0.0f)
                return BSDFSample{float3(0.0f), float3(0.0f), 1.0f};
            
            float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
            float cosON = abs(dot(wo, n));
            
            float3 H = normalize(wi + wo);
            float3 localH = normalize(float3(dot(H, T), dot(H, B), dot(H, n)));
            float cosIH = abs(dot(wi, H));
            float fresnel_R = dielectricFresnel(cosIH, eta);
            
            float D = D_GGX(localH, alpha, alpha);
            float G = G_Smith(wiLocal, woLocal, alpha);
            float G1 = G1_Smith(wiLocal, alpha);
            
            float3 specBSDF = float3(fresnel_R) * D * G / (4.0f * cosIN * cosON);
            float3 diffBSDF = (1.0f - fresnel_R) * diffuseBRDF(wi, wo, n, material);
            
            float specPDF = (D * G1) / (4.0f * cosIN);
            float diffPDF = diffusePDF(wi, wo, n);
            float PDF = pr * specPDF + pd * diffPDF;
            
            return BSDFSample{specBSDF + diffBSDF, wo, PDF, false, false};
        }
    }
}

float3 dielectricBSDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    if (material.roughness < 0.01f) {
        if (material.BXDFs == DIELECTRIC_REFLECTION) {
            float cosIN = dot(wi, n);
            bool entering = cosIN > 0.0f;
            n = entering ? n : -n;
            
            float eta = entering ? 1.0f / material.ior : material.ior;
            float cosON = dot(wo, n);
            
            if (cosON <= 0.0f)
                return float3(0.0f);
            
            float fresnel_R = dielectricFresnel(abs(cosIN), eta);
            return (1.0f - fresnel_R) * diffuseBRDF(wi, wo, n, material);
        }
        
        return float3(0.0f);
    }

    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    
    float eta = entering ? 1.0f / material.ior : material.ior;
    cosIN = abs(cosIN);
    float cosON = dot(wo, n);
    
    bool allowTransmission = material.BXDFs == DIELECTRIC_TRANSMISSION;
    bool sameHemisphere = cosON > 0.0f;

    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float alpha = material.roughness * material.roughness;

    float3 BSDF = float3(0.0f);

    if (sameHemisphere) {
        float3 wm = normalize(wiLocal + woLocal);
        float cosIH = abs(dot(wiLocal, wm));
        float fresnel_R = dielectricFresnel(cosIH, eta);
        float D = D_GGX(wm, alpha, alpha);
        float G = G_Smith(wiLocal, woLocal, alpha);
        BSDF += float3(fresnel_R) * D * G / (4.0f * cosIN * abs(cosON));

        if (!allowTransmission)
            BSDF += (1.0f - fresnel_R) * diffuseBRDF(wi, wo, n, material);
    }

    if (!sameHemisphere && allowTransmission) {
        float3 wm = normalize(wiLocal * eta + woLocal);
        float cosIH = dot(wiLocal, wm);
        float cosOH = dot(woLocal, wm);
        
        float fresnel_T = 1.0f - dielectricFresnel(abs(cosIH), eta);
        float D = D_GGX(wm, alpha, alpha);
        float G = G_Smith(wiLocal, woLocal, alpha);
        float denom = eta * cosIH + cosOH;
        float jacobian = eta * eta * abs(cosIH * cosOH) / (denom * denom);
        
        BSDF += float3(fresnel_T) * D * G * jacobian / (cosIN * abs(cosON));
    }

    return BSDF;
}

float dielectricPDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    if (material.roughness < 0.01f) {
        if (material.BXDFs == DIELECTRIC_REFLECTION) {
            float cosIN = dot(wi, n);
            bool entering = cosIN > 0.0f;
            n = entering ? n : -n;
            float cosON = dot(wo, n);
            
            if (cosON <= 0.0f)
                return 0.0f;
            
            float eta = entering ? 1.0f / material.ior : material.ior;
            float fresnel_R = dielectricFresnel(abs(cosIN), eta);
            
            return (1.0f - fresnel_R) * diffusePDF(wi, wo, n);
        }
        
        return 0.0f;
    }

    float cosIN = dot(wi, n);
    bool entering = cosIN > 0.0f;
    n = entering ? n : -n;
    
    float eta = entering ? 1.0f / material.ior : material.ior;
    cosIN = abs(cosIN);
    float cosON = dot(wo, n);
    
    bool allowTransmission = material.BXDFs == DIELECTRIC_TRANSMISSION;
    bool sameHemisphere = cosON > 0.0f;

    float3 T, B;
    createOrthonormalBasis(n, T, B);
    float3 wiLocal = float3(dot(wi, T), dot(wi, B), dot(wi, n));
    float3 woLocal = float3(dot(wo, T), dot(wo, B), dot(wo, n));
    float alpha = material.roughness * material.roughness;

    float F0 = ((material.ior - 1.0f) / (material.ior + 1.0f));
    F0 = F0 * F0;
    float avgR = F0 + (1.0f - F0) * 0.15f;

    float pr = allowTransmission ? 0.5f : avgR;
    float pt = allowTransmission ? 0.5f : 0.0f;
    float pd = allowTransmission ? 0.0f : 1.0f - avgR;

    float PDF = 0.0f;

    if (sameHemisphere) {
        float3 wm = normalize(wiLocal + woLocal);
        float D = D_GGX(wm, alpha, alpha);
        float G1 = G1_Smith(wiLocal, alpha);
        PDF += pr * (D * G1) / (4.0f * cosIN);

        if (!allowTransmission)
            PDF += pd * diffusePDF(wi, wo, n);
    }

    if (!sameHemisphere && allowTransmission) {
        float3 wm = normalize(wiLocal * eta + woLocal);
        float cosIH = dot(wiLocal, wm);
        float cosOH = dot(woLocal, wm);
        float D = D_GGX(wm, alpha, alpha);
        float G1 = G1_Smith(wiLocal, alpha);
        float denom = eta * cosIH + cosOH;
        float dwm_dwi = abs(cosOH) / (denom * denom);
        PDF += pt * D * G1 * abs(cosIH) / cosIN * dwm_dwi;
    }

    return PDF;
}

// MARK: Dispatch

BSDFSample sampleBXDF(float3 wi, float3 n, SampledMaterial material, float3 r3) {
    BSDFSample bsdfSample;
    
    if (material.BXDFs == DIFFUSE) {
        bsdfSample = sampleDiffuseBRDF(wi, n, material, r3.xy);
    } else if (material.BXDFs == CONDUCTOR) {
        bsdfSample = sampleConductorBRDF(wi, n, material, r3.xy);
    } else if (material.BXDFs & (DIELECTRIC_TRANSMISSION | DIELECTRIC_REFLECTION)) {
        bsdfSample = sampleDielectricBSDF(wi, n, material, r3.x, r3.yz);
    } else {
        DEBUG("sampleBXDF - BXDF not found. BXDF: %d", material.BXDFs);
        bsdfSample = BSDFSample{float3(0.0f), float3(0.0f), 1.0f};
    }
    
    bsdfSample.BSDF = max(bsdfSample.BSDF, float3(0.0f));
    if (bsdfSample.PDF <= 0.0f) {
        bsdfSample.BSDF = float3(0.0f);
        bsdfSample.PDF = 1.0f;
    }
        
    return bsdfSample;
}

float3 getBXDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    float3 BXDF;
    
    if (material.BXDFs == DIFFUSE) {
        BXDF = diffuseBRDF(wi, wo, n, material);
    } else if (material.BXDFs == CONDUCTOR) {
        BXDF = conductorBSDF(wi, wo, n, material);
    } else if (material.BXDFs & (DIELECTRIC_TRANSMISSION | DIELECTRIC_REFLECTION)) {
        BXDF = dielectricBSDF(wi, wo, n, material);
    } else {
        DEBUG("getBXDF - BXDF not found. BXDF: %d", material.BXDFs);
        BXDF = float3(0.0f);
    }

    return max(BXDF, float3(0.0f));
}

float getPDF(float3 wi, float3 wo, float3 n, SampledMaterial material) {
    float PDF;

    if (material.BXDFs == DIFFUSE) {
        PDF = diffusePDF(wi, wo, n);
    } else if (material.BXDFs == CONDUCTOR) {
        PDF = conductorPDF(wi, wo, n, material);
    } else if (material.BXDFs & (DIELECTRIC_TRANSMISSION | DIELECTRIC_REFLECTION)) {
        PDF = dielectricPDF(wi, wo, n, material);
    } else {
        DEBUG("getPDF - BXDF not found. BXDF: %d", material.BXDFs);
        PDF = 0.0f;
    }

    return max(PDF, 0.0f);
}
