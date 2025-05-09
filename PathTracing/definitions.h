//
//  definitions.h
//  PathTracing
//
//  Created on 3/21/25.
//

#ifndef definitions_h
#define definitions_h

#include <simd/simd.h>

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_SPHERE   2
#define GEOMETRY_MASK_LIGHT    4
#define GEOMETRY_MASK_TRANSPARENT 8
#define GEOMETRY_MASK_OPAQUE 16

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_SPHERE)

#define RAY_MASK_PRIMARY   (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_TRANSPARENT | GEOMETRY_MASK_OPAQUE)
#define RAY_MASK_SHADOW    GEOMETRY_MASK_OPAQUE
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_TRANSPARENT | GEOMETRY_MASK_OPAQUE

#define MAX_PATH_LENGTH 6
#define MAX_CAMERA_PATH_LENGTH (MAX_PATH_LENGTH + 2)
#define MAX_LIGHT_PATH_LENGTH (MAX_PATH_LENGTH + 1)

#define CAMERA_VERTEX 1
#define LIGHT_VERTEX 2

#define MAX_AREA_LIGHTS 16

#define CAMERA_FOV_ANGLE 60

#define MAX_TEXTURES 120

struct Camera {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
};

struct LightTriangle {
    vector_float3 v0;
    vector_float3 v1;
    vector_float3 v2;
    vector_float3 emission0;
    vector_float3 emission1;
    vector_float3 emission2;
    float area;
    float cdf;
};

struct AreaLight {
    vector_float3 position;
    vector_float3 forward;
    vector_float3 right;
    vector_float3 up;
    vector_float3 color;
    unsigned int firstTriangleIndex;
    unsigned int triangleCount;
    float totalArea;
    simd_float4x4 transform;
};

struct Uniforms {
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    struct Camera camera;
    unsigned int lightCount;
};

struct Sphere {
    vector_float3 origin;
    float radius;
    vector_float3 color;
};

struct Material {
    float opacity;
    float refraction;
    float roughness;
    float metallic;
    unsigned int texture_index;
};

struct PathVertex {
    vector_float3 position;
    vector_float3 normal;
    vector_float3 throughput;
    vector_float3 material_color;
    vector_float3 incoming_direction;
    struct Material material;
    float forwardPDF;
    float reversePDF;
    int is_delta;
    int type;
};

struct LightSample {
    struct AreaLight areaLight;
    vector_float3 position;
    vector_float3 direction;
    float distance;
    vector_float3 normal;
    vector_float3 emission;
    float PDF;
};

#endif
