//
//  Integrators.h
//  PathTracing
//
//  Created on 7/19/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"

#define MAX_PATH_LENGTH 5
#define MAX_CAMERA_PATH_LENGTH (MAX_PATH_LENGTH + 2)
#define MAX_LIGHT_PATH_LENGTH (MAX_PATH_LENGTH + 1)

enum VertexType : unsigned int {
    CAMERA_VERTEX = 0,
    LIGHT_VERTEX = 1,
    SURFACE_VERTEX = 2
};

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

float3 pathIntegrator(float2 pixel,
                      constant Uniforms& uniforms,
                      constant unsigned int& resourceStride,
                      device void *resources,
                      device MTLAccelerationStructureInstanceDescriptor *instances,
                      instance_acceleration_structure accelerationStructure,
                      constant Light *lights,
                      constant LightTriangle *lightTriangles,
                      constant int *lightIndices,
                      texture2d<float> environmentMapTexture,
                      constant float *environmentMapCDF,
                      array<texture2d<float>, MAX_TEXTURES> textureArray,
                      thread HaltonSampler& haltonSampler
                      );

float3 bidirectionalPathIntegrator(float2 pixel,
                                   constant Uniforms& uniforms,
                                   constant unsigned int& resourceStride,
                                   device void *resources,
                                   device MTLAccelerationStructureInstanceDescriptor *instances,
                                   instance_acceleration_structure accelerationStructure,
                                   constant Light *lights,
                                   constant LightTriangle *lightTriangles,
                                   constant int *lightIndices,
                                   texture2d<float> environmentMapTexture,
                                   constant float *environmentMapCDF,
                                   array<texture2d<float>, MAX_TEXTURES> textureArray,
                                   thread HaltonSampler& haltonSampler,
                                   texture2d<float, access::read_write> splatTex,
                                   device atomic_float* splatBuffer
                                   );

struct EndpointInteraction {
    float3 position;
    float3 normal;
    union {
        constant Camera* camera;
        constant Light* light;
    };
    
    EndpointInteraction(float3 _position, float3 _normal, constant Camera* _camera) {
        position = _position;
        normal = _normal;
        camera = _camera;
    }
    
    EndpointInteraction(float3 _position, float3 _normal, constant Light* _light) {
        position = _position;
        normal = _normal;
        light = _light;
    }
    
    EndpointInteraction() {
        
    }
};

struct PathVertex {
    enum VertexType type;
    SurfaceInteraction si;
    EndpointInteraction ei;
    float3 throughput;
    float forwardPDF;
    float reversePDF;
    bool delta;
    
    PathVertex(VertexType _type, SurfaceInteraction _interaction, float3 _throughput, float _forwardPDF = 0.0f, float _reversePDF = 0.0f, bool _delta = false) {
        type = _type;
        throughput = _throughput;
        si = _interaction;
        forwardPDF = _forwardPDF;
        reversePDF = _reversePDF;
        delta = _delta;
    }
    
    PathVertex(VertexType _type, EndpointInteraction _interaction, float3 _throughput, float _forwardPDF = 0.0f, float _reversePDF = 0.0f, bool _delta = false) {
        type = _type;
        throughput = _throughput;
        ei = _interaction;
        forwardPDF = _forwardPDF;
        reversePDF = _reversePDF;
        delta = _delta;
    }

    PathVertex() {
        forwardPDF = 0.0f;
        reversePDF = 0.0f;
        delta = false;
    }
    
    float convertDensity(float PDF, thread PathVertex& nxt) {
        if (nxt.isInfiniteLight())
            return PDF;
        
        float3 w = nxt.position() - position();
        float d2_inv = 1.0f / length_squared(w);
        if (nxt.isOnSurface())
            PDF *= abs(dot(nxt.normal(), w * d2_inv));

        return PDF * d2_inv;
    }
    
    float3 BXDF(float3 wi, PathVertex nxt) {
        float3 wo = normalize(nxt.position() - position());

        switch (type) {
            case SURFACE_VERTEX:
                return getBXDF(wi, wo, normal(), si.material);
            default:
                DEBUG("Vertex BXDF unimplemented.");
                return float3(0.0f);
        }
    }
    
    float PDF(thread PathVertex& prev, thread PathVertex& nxt, constant float* environmentMapCDF) {
        if (type == LIGHT_VERTEX) {
            if (!ei.light) DEBUG("NO LIGHT");
            return lightDirectionPDF(*ei.light, nxt, environmentMapCDF);
        }
        
        float3 wo = normalize(nxt.position() - position());
        
        float PDF, unused;
        if (type == CAMERA_VERTEX) {
            PDF = 0.0f;
//            cameraRayPDF(*ei.camera, wo, unused, PDF);
        } else if (type == SURFACE_VERTEX) {
            float3 wi = normalize(position() - prev.position());
            PDF = getPDF(-wi, wo, normal(), si.material);
        } else {
            DEBUG("Vertex PDF unimplemented.");
            PDF = 1.0f;
        }
        
        return convertDensity(PDF, nxt);
    }
        
    
    float lightOriginPDF(constant Light& light, constant Light *lights, constant Uniforms& uniforms, thread PathVertex& nxt, constant float* environmentMapCDF) {
        float3 w = normalize(nxt.position() - position());
        
        if (isInfiniteLight()) {
            return infiniteLightDensity(-w, lights, uniforms, environmentMapCDF);
        }

        float selectionPDF = getLightSelectionPDF(light, lights, uniforms);
        float positionPDF = getLightSamplePDF(light);
        return selectionPDF * positionPDF;
    }
    
    float lightDirectionPDF(constant Light& light, thread PathVertex& nxt, constant float* environmentMapCDF) {
        float3 wo = nxt.position() - position();
        float d2_inv = 1.0f / length_squared(wo);
        wo *= sqrt(d2_inv);

        float directionPDF;
        
        if (isInfiniteLight()) {
            directionPDF = 1.0f / (M_PI_F * SCENE_RADIUS * SCENE_RADIUS);
        } else {
            directionPDF = d2_inv * getLightDirectionPDF(light, wo, normal(), environmentMapCDF);
        }
                
        if (nxt.isOnSurface())
            directionPDF *= abs(dot(nxt.normal(), wo));
        
        return directionPDF;
    }
    
    float3 getLightEmission(thread PathVertex& prev, constant Light* lights, texture2d<float> environmentMapTexture) {
        if (!isLight())
            return float3(0.0f);
        
        float3 w = normalize(position() - prev.position());
        
        if (type == LIGHT_VERTEX) {
            float2 uv = getEnvironmentMapUV(w);
            return environmentMapEmission(uv, environmentMapTexture);
        } else if (si.hitLight) {
            return lights[si.lightIndex].color;
        }
        
        return float3(0.0f);
    }
    
    inline bool isOnSurface() {
        return type != CAMERA_VERTEX && length_squared(normal()) > 1e-1f;
    }
    
    inline bool isConnectible() {
        switch (type) {
            case CAMERA_VERTEX:
                return true;
            case LIGHT_VERTEX:
                return ei.light->type != DIRECTIONAL_LIGHT;
            case SURFACE_VERTEX:
                return !delta;
            default:
                DEBUG("Vertex type not found.");
                return false;
        }
    }
    
    inline bool isLight() {
        return type == LIGHT_VERTEX || (type == SURFACE_VERTEX && si.hitLight);
    }
    
    inline bool isInfiniteLight() {
        return type == LIGHT_VERTEX && (!ei.light || ei.light->type == ENVIRONMENT_MAP/* || ei.light->type == DIRECTIONAL_LIGHT*/);
    }
    
    inline bool isDeltaLight() {
        return type == LIGHT_VERTEX && (ei.light && ei.light->delta);
    }
            
    inline float3 position() {
        switch (type) {
            case CAMERA_VERTEX:
                return ei.position;
            case LIGHT_VERTEX:
                return ei.position;
            case SURFACE_VERTEX:
                return si.position;
            default:
                DEBUG("Vertex not handled in position.");
                return float3(0.0f);
        }
    }
    
    inline float3 normal() {
        switch (type) {
            case CAMERA_VERTEX:
                return ei.normal;
            case LIGHT_VERTEX:
                return ei.light ? ei.normal : float3(0.0f);
            case SURFACE_VERTEX:
                return si.normal;
            default:
                DEBUG("Vertex not handled in position.");
                return float3(0.0f);
        }
    }
};

inline PathVertex createSurfaceVertex(SurfaceInteraction interaction, float3 throughput, float PDF, PathVertex prev) {
    PathVertex vx = PathVertex(SURFACE_VERTEX, interaction, throughput);
    vx.forwardPDF = prev.convertDensity(PDF, vx);
    return vx;
}

inline PathVertex createCameraVertex(constant Camera* camera, float3 position, float3 normal, float3 throughput) {
    return PathVertex(CAMERA_VERTEX, EndpointInteraction(position, normal, camera), throughput);
}

inline PathVertex createLightVertex(constant Light* light, float3 position, float3 normal, float3 emission, float forwardPDF) {
    return PathVertex(LIGHT_VERTEX, EndpointInteraction(position, normal, light), emission, forwardPDF);
}

#endif
