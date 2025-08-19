//
//  Integrators.h
//  PathTracing
//
//  Created by Moritz Kohlenz on 7/19/25.
//

#pragma once

#include <simd/simd.h>
#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"

#define MAX_PATH_LENGTH 6
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
                      HaltonSampler haltonSampler
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
                                   HaltonSampler haltonSampler
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
        
    }
    
    float convertDensity(float PDF, thread PathVertex& nxt) {
        float3 w = nxt.position() - position();
        float d2_inv = 1.0f / length_squared(w);
        float cosN = abs(dot(nxt.normal(), w * d2_inv));
        return PDF * cosN * d2_inv;
    }
    
    float3 BXDF(float3 wi, PathVertex nxt) {
        float3 wo = normalize(nxt.position() - position());
        
        switch (type) {
            case SURFACE_VERTEX:
                return getBXDF(wi, wo, normal(), si.material);
            default:
                unimplemented();
                return float3(0.0f);
        }
    }
    
    float PDF(thread PathVertex& prev, thread PathVertex& nxt) {
        if (type == LIGHT_VERTEX) {
            return lightDirectionPDF(*ei.light, nxt);
        }
        
        float3 wo = normalize(nxt.position() - position());
        
        float PDF, unused;
        if (type == CAMERA_VERTEX) {
            cameraRayPDF(*ei.camera, -wo, unused, PDF);
        } else if (type == SURFACE_VERTEX) {
            float3 wi = normalize(position() - prev.position());
            PDF = getPDF(-wi, wo, normal(), si.material);
        } else {
            unimplemented();
            PDF = 0.0f;
        }
        
        return convertDensity(PDF, nxt);
    }
        
    
    float lightOriginPDF(constant Light& light, constant Light *lights, constant Uniforms& uniforms) {
        float selectionPDF = getLightSelectionPDF(light, lights, uniforms);
        float positionPDF = getLightSamplePDF(light);
        return selectionPDF * positionPDF;
    }
    
    float lightDirectionPDF(constant Light& light, thread PathVertex& nxt) {
        float3 wo = nxt.position() - position();
        float d2_inv = 1.0f / length_squared(wo);
        wo *= sqrt(d2_inv);
        
        float directionPDF = getLightDirectionPDF(light, wo, normal());

        if (nxt.isOnSurface())
            directionPDF *= abs(dot(nxt.normal(), wo));
        
        return directionPDF * d2_inv;
    }
    
    inline bool isOnSurface() {
        return length(normal()) > 1e-1f;
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
                return ei.normal;
            case SURFACE_VERTEX:
                return si.normal;
            default:
                DEBUG("Vertex not handled in position.");
                return float3(0.0f);
        }
    }

    inline thread SurfaceInteraction& getSurfaceInteraction() thread {
        return si;
    }
    
    inline thread EndpointInteraction& getEndpointInteraction() thread {
        return ei;
    }
};

inline float convertDensity(float PDF, PathVertex vx, PathVertex nxt) {
    float3 w = nxt.position() - vx.position();
    float d2_inv = 1.0f / length_squared(w);
    float cosN = abs(dot(nxt.normal(), w * d2_inv));
    return PDF * cosN * d2_inv;
}

inline PathVertex createSurfaceVertex(SurfaceInteraction interaction, float3 throughput, float PDF, PathVertex prev) {
    PathVertex vx = PathVertex(SURFACE_VERTEX, interaction, throughput);
    vx.forwardPDF = convertDensity(PDF, prev, vx);
    return vx;
}

inline PathVertex createCameraVertex(constant Camera& camera, float3 position, float3 normal, float3 throughput) {
    return PathVertex(CAMERA_VERTEX, EndpointInteraction(position, normal, &camera), throughput);
}

inline PathVertex createLightVertex(constant Light& light, float3 position, float3 normal, float3 emission, float forwardPDF) {
    return PathVertex(LIGHT_VERTEX, EndpointInteraction(position, normal, &light), emission, forwardPDF);
}

#endif
