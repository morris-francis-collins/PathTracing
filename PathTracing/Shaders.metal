//
//  Shaders.metal
//  PathTracing
//

#include "definitions.h"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
using namespace raytracing;

constant unsigned int resourcesStride  [[function_constant(0)]];
constant bool useIntersectionFunctions [[function_constant(1)]];

constant unsigned int primes[] =
{
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449,
    457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557
};

// some functions from original template code
// |
// v

// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base.  The Halton sequence is a "low discrepency" sequence: the values appear
// random but are more evenly distributed than a purely random sequence.  Each random
// value used to render the image should use a different independent dimension `d`,
// and each sample (frame) should use a different index `i`. To decorrelate each
// pixel, you can apply a random offset to 'i'.
float halton(unsigned int i, unsigned int d)
{
    unsigned int b = primes[d];

    float f = 1.0f;
    float invB = 1.0f / b;

    float r = 0;

    while (i > 0)
    {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }
    
    return r;
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection structure.
template<typename T>
inline T interpolateVertexAttribute(device T *attributes, unsigned int primitiveIndex, float2 uv)
{
    // Look up value for each vertex.
    T T0 = attributes[primitiveIndex * 3 + 0];
    T T1 = attributes[primitiveIndex * 3 + 1];
    T T2 = attributes[primitiveIndex * 3 + 2];

    // Compute sum of vertex attributes weighted by barycentric coordinates.
    // Barycentric coordinates sum to one.
    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

// Uses the inversion method to map two uniformly random numbers to a three-dimensional
// unit hemisphere, where the probability of a given sample is proportional to the cosine
// of the angle between the sample direction and the "up" direction (0, 1, 0).
inline float3 sampleCosineWeightedHemisphere(float2 u)
{
    float phi = 2.0f * M_PI_F * u.x;

    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    

    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

// Aligns a direction on the unit hemisphere such that the hemisphere's "up" direction
// (0, 1, 0) maps to the given surface normal direction.
inline float3 alignHemisphereWithNormal(float3 sample, float3 normal)
{
    // Set the "up" vector to the normal
    float3 up = normal;

    // Find an arbitrary direction perpendicular to the normal. This will become the
    // "right" vector.
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));

    // Find a third vector perpendicular to the previous two. This will be the
    // "forward" vector.
    float3 forward = cross(right, up);

    // Map the direction on the unit hemisphere to the coordinate system aligned
    // with the normal.
    return sample.x * right + sample.y * up + sample.z * forward;
}

// Return type for a bounding box intersection function.
struct BoundingBoxIntersection
{
    bool accept; // Whether to accept or reject the intersection.
    float distance; // Distance from the ray origin to the intersection point.
};

typedef BoundingBoxIntersection IntersectionFunction(float3,
                                                     float3,
                                                     float,
                                                     float,
                                                     unsigned int,
                                                     unsigned int,
                                                     device void *);

__attribute__((always_inline))
float3 transformPoint(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 1.0f);
}

__attribute__((always_inline))
float3 transformDirection(float3 p, float4x3 transform) {
    return transform * float4(p.x, p.y, p.z, 0.0f);
}

// Reuse the standard ray tracing API's intersection result type since it has storage
// for all of the data necessary.
typedef intersector<triangle_data, instancing, world_space_data>::result_type IntersectionResult;

// Intersects a ray with an acceleration structure, dispatching calls to intersection
// functions if needed.
IntersectionResult intersect(ray ray,
                             unsigned int mask,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             instance_acceleration_structure accelerationStructure,
                             visible_function_table<IntersectionFunction> intersectionFunctionTable,
                             bool accept_any_intersection)
{
    // Parameters used to configure the intersection query.
    intersection_params params;
    
    // Create an intersection query to test for intersection between the ray and the geometry
    // in the scene.  The `intersection_query` object tracks the current state of the acceleration
    // structure traversal.
    intersection_query<triangle_data, instancing> i;
    
    // If the sample isn't using intersection functions, provide some hints to Metal for
    // better performance.
    if (!useIntersectionFunctions)
    {
        params.assume_geometry_type(geometry_type::triangle);
        params.force_opacity(forced_opacity::opaque);
    }
    
    // Shadow rays check only whether there is an object between the intersection point
    // and the light source. In that case, tell Metal to return after finding any intersection.
    params.accept_any_intersection(accept_any_intersection);

    // Initialize the intersection query with the ray, acceleration structure, and other parameters.
    i.reset(ray, accelerationStructure, mask, params);

    if (!useIntersectionFunctions)
    {
        // No bounding box intersection will be reported if the acceleration structure does
        // not contain bounding box primitives, so the entire acceleration structure is
        // traversed using a single call to `next()`.
        i.next();
    }
    else
    {
        // Otherwise, we will need to handle bounding box intersections as they are found. Call
        // `next()` in a loop until it returns `false`, indicating that acceleration structure traversal
        // is complete.
        while (i.next())
        {
            // The intersection query object keeps track of the "candidate" and "committed"
            // intersections. The "committed" intersection is the current closest intersection
            // found, while the "candidate" intersection is a potential intersection. Dispatch
            // a call to the corresponding intersection function to determine whether to accept
            // the candidate intersection.
            unsigned int instanceIndex = i.get_candidate_instance_id();
            
            unsigned int resourceIndex = instances[instanceIndex].accelerationStructureIndex;
            
            BoundingBoxIntersection bb = intersectionFunctionTable[resourceIndex](
                                             // Ray origin and direction in object space for the candidate instance.
                                             i.get_candidate_ray_origin(),
                                             i.get_candidate_ray_direction(),
                                             // Minimum and maximum intersection distance to consider.
                                             i.get_ray_min_distance(),
                                             i.get_committed_distance(),
                                             // Information about candidate primitive.
                                             i.get_candidate_primitive_id(),
                                             resourceIndex,
                                             resources);
            
            // Accept the candidate intersection, making it the new committed intersection.
            if (bb.accept)
                i.commit_bounding_box_intersection(bb.distance);
        }
    }
    
    IntersectionResult intersection;
    
    // Return all the information about the committed intersection.
    intersection.type = i.get_committed_intersection_type();
    intersection.distance = i.get_committed_distance();
    intersection.primitive_id = i.get_committed_primitive_id();
    intersection.geometry_id = i.get_committed_geometry_id();
    intersection.triangle_barycentric_coord = i.get_committed_triangle_barycentric_coord();
    intersection.instance_id = i.get_committed_instance_id();
    intersection.object_to_world_transform = i.get_committed_object_to_world_transform();
    
    return intersection;
}

uint hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float scrambledHalton(uint index, uint dimension, uint frameIndex) {
    float value = halton(index, dimension);
    return fract(value + float(hash(dimension + frameIndex * 719)) / 4294967296.0);
}

LightSample sampleTriangle(thread AreaLight areaLight,
                      device LightTriangle *lightTriangles,
                      float selectionPDF,
                      ray ray,
                      float3 r
                      )
{
    LightSample sample;

    int left = areaLight.firstTriangleIndex;
    int right = areaLight.firstTriangleIndex + areaLight.triangleCount - 1;
    float target = r.x;

    while (left < right) {
        int mid = (left + right) / 2;

        if (target < lightTriangles[mid].cdf) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    LightTriangle triangle = lightTriangles[left];
    if (r.y + r.z > 1.0f) {
        r.y = 1 - r.y;
        r.z = 1 - r.z;
    }

    float u = 1.0f - r.y - r.z;
    float v = r.y;
    float w = r.z;
    
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    sample.position = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    sample.normal = normalize(cross(edge1, edge2));

    float3 toLight = sample.position - ray.origin;
    sample.distance = max(length(toLight), 1e-4f);
    sample.direction = toLight / sample.distance;

    sample.emission = u * triangle.emission0 + v * triangle.emission1 + w * triangle.emission2;
    sample.PDF = selectionPDF / triangle.area;
    return sample;
}

LightSample sampleAreaLight(device AreaLight *areaLights,
                       device LightTriangle *lightTriangles,
                       constant Uniforms& uniforms,
                       float3 point,
                       float3 direction,
                       float4 r
                       )
{
    LightSample sample;
    float weights[MAX_AREA_LIGHTS];
    float totalWeight = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        AreaLight light = areaLights[i];
        
        float3 toLight = light.position - point;
        float distanceSq = max(1e-3f, dot(toLight, toLight));
        float distance = sqrt(distanceSq);
        float3 toLightNorm = toLight / distance;
        
        
        float cosTheta = max(1e-3f, dot(direction, toLightNorm));
        float solidAngleFactor = light.totalArea / (distance * distance);
        float intensity = length(light.color);
                
        weights[i] = cosTheta * solidAngleFactor * intensity;
        totalWeight += weights[i];
    }
    
    if (totalWeight < 1e-4f) {
        sample = sampleTriangle(areaLights[0], lightTriangles, 1.0f / float(uniforms.lightCount), point, r.yzw);
        sample.areaLight = areaLights[0];
        return sample;
    }
    
    float random = r.x * totalWeight;
    float accumWeight = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        accumWeight += weights[i];
        if (accumWeight >= random) {
            sample = sampleTriangle(areaLights[i], lightTriangles, weights[i] / totalWeight, point, r.yzw);
            sample.areaLight = areaLights[i];
            return sample;
        }
    }
    
    sample = sampleTriangle(areaLights[uniforms.lightCount - 1], lightTriangles, weights[uniforms.lightCount - 1] / totalWeight, point, r.yzw);
    sample.areaLight = areaLights[uniforms.lightCount - 1];
    return sample;
}

float3 evaluateBSDF(PathVertex vx, float3 outgoingDirection) {
    Material material = vx.material;
    float3 incomingDirection = vx.incoming_direction;
    float3 normal = vx.normal;
    
    float cosIncoming = dot(incomingDirection, normal);
    float cosOutgoing = dot(outgoingDirection, normal);
    
    bool sameHemisphere = (cosIncoming * cosOutgoing) > 0;
    
    float dielectricF0 = pow((1.0f - material.refraction) / (1.0f + material.refraction), 2.0f);
    float reflectance = mix(dielectricF0, 0.95f, material.metallic);
    
    if (material.opacity < 1.0f) {
        if (!sameHemisphere) {
            float transmittance = 1.0f - reflectance;
            return float3(transmittance) * (1.0f - material.opacity);
        } else {
            if (material.roughness < 0.05f) {
                return float3(reflectance) * vx.material_color;
            } else {
                // rough reflection
                float roughness2 = material.roughness * material.roughness;
                float3 halfVector = normalize(incomingDirection + outgoingDirection);
                float NdotH = saturate(dot(normal, halfVector));
                
                // microfacet model
                float D = roughness2 / (M_PI_F * pow(NdotH * NdotH * (roughness2 - 1.0f) + 1.0f, 2.0f));
                float G = cosIncoming * cosOutgoing;
                
                return float3(reflectance) * vx.material_color * D * G / (4.0f * abs(cosIncoming) * abs(cosOutgoing));
            }
        }
    }
    else if (material.metallic > 0.5f) {
        if (material.roughness < 0.05f) {
            return float3(0.0f);
        } else {
            float roughness2 = material.roughness * material.roughness;
            float3 halfVector = normalize(incomingDirection + outgoingDirection);
            float NdotH = saturate(dot(normal, halfVector));
            
            // GGX
            float D = roughness2 / (M_PI_F * pow(NdotH * NdotH * (roughness2 - 1.0f) + 1.0f, 2.0f));
            float G = cosIncoming * cosOutgoing;
            
            return  float3(reflectance) * vx.material_color * D * G / (4.0f * abs(cosIncoming) * abs(cosOutgoing));
        }
    }
    else {
        // lambertian diffuse
        if (sameHemisphere) {
            float dielectricReflectance = dielectricF0 + (1.0f - dielectricF0) * pow(1.0f - abs(cosOutgoing), 5.0f);
            float diffuseFactor = (1.0f - dielectricReflectance) * (1.0f - material.metallic);
            return vx.material_color * diffuseFactor / M_PI_F;
        } else {
            return float3(0.0f);
        }
    }
}

struct TriangleResources
{
    device float3 *vertexNormals;
    device float3 *vertexColors;
    device Material *vertexMaterials;
    device float2 *vertexUVs;
};

struct BounceInfo {
    ray ray;
    int is_delta;
    float3 normal;
};

float calculateReflectance(Material material) {
    if (material.metallic > 0.5f) {
        return 0.9f;
    }
    
    float F0 = pow((1.0f - material.refraction) / (1.0f + material.refraction), 2.0f);
    float roughnessEffect = 1.0f - material.roughness;
    
    return F0 * roughnessEffect;
}

BounceInfo calculateBounce(ray incidentRay,
                           float3 intersectionPoint,
                           float3 normal,
                           Material material,
                           uint offset,
                           uint bounce,
                           uint frameIndex
                           )
{
    BounceInfo bounceInfo;
    bounceInfo.ray.max_distance = INFINITY;
    bounceInfo.is_delta = false;
    
    float2 r = float2(scrambledHalton(offset, 5 + bounce * 13 + 0, frameIndex),
                      scrambledHalton(offset, 5 + bounce * 13 + 1, frameIndex));
    float random = scrambledHalton(offset, 5 + bounce * 13 + 2, frameIndex);
    
    float dielectricF0 = pow((1.0f - material.refraction) / (1.0f + material.refraction), 2.0f);
    float reflectance = mix(dielectricF0, 0.95f, material.metallic);
    float3 reflected = reflect(incidentRay.direction, normal);
    
    float3 n = normal;
    
    if (material.opacity < 1.0f && random > material.opacity) {
        bool entering = dot(incidentRay.direction, n) < 0.0f;
        n = entering ? n : -n;
        float eta = entering ? (1.0f / material.refraction) : material.refraction;
        float cosTheta = saturate(dot(-incidentRay.direction, n));
        float fresnel = dielectricF0 + (1.0f - dielectricF0) * pow(1.0f - cosTheta, 5.0f);
        
        float3 refracted = refract(incidentRay.direction, n, eta);
        
        if (r.x > fresnel) { // refract
            if (length(refracted) < 1e-5f) {
                refracted = reflected;
            } // add opacity scaling with beers law
            
            bounceInfo.ray.direction = refracted;

        } else { // reflect
            bounceInfo.ray.direction = reflected;
        }
        
        bounceInfo.ray.origin = intersectionPoint - n * 1e-5f;
    } else {
        if (material.metallic > 0.5f || random < reflectance) {
            float3 reflected = reflect(incidentRay.direction, n);
            
            if (material.roughness > 0.05f) {
                float3 randomDir = sampleCosineWeightedHemisphere(r);
                randomDir = alignHemisphereWithNormal(randomDir, n);
                bounceInfo.ray.direction = normalize(mix(reflected, randomDir, material.roughness));
            } else {
                bounceInfo.ray.direction = reflected;
            }
        } else {
            float3 diffuseDir = sampleCosineWeightedHemisphere(r);
            bounceInfo.ray.direction = alignHemisphereWithNormal(diffuseDir, n);
        }
        bounceInfo.ray.origin = intersectionPoint + n * 1e-5f;
    }
    
    if (material.opacity < 1.0f) {
        bounceInfo.is_delta = (material.roughness < 0.2f);
    } else if (material.metallic > 0.5f) {
        bounceInfo.is_delta = (material.roughness < 0.2f);
    } else {
        bounceInfo.is_delta = false;
    }
    
    bounceInfo.ray.direction = normalize(bounceInfo.ray.direction);
    bounceInfo.normal = n;
    return bounceInfo;
}

void calculateNewPDFs(thread PathVertex& currVertex,
                      thread PathVertex& prevVertex,
                      float3 direction, // of new ray after bounce
                      float3 n
                      )
{
    if (currVertex.is_delta) {
        currVertex.forwardPDF = 1.0f;
        currVertex.reversePDF = 0.0f;
    } else {
        float cosTheta = max(0.01f, abs(dot(direction, n)));
        currVertex.forwardPDF = cosTheta / M_PI_F;
        
        float3 toPrev = normalize(prevVertex.position - currVertex.position);
        float cosToPrev = abs(dot(toPrev, n));
        currVertex.reversePDF = cosToPrev / M_PI_F;
    }
}

int traceCameraPath(float2 pixel,
                    constant Uniforms & uniforms,
                    unsigned int offset,
                    device void *resources,
                    device MTLAccelerationStructureInstanceDescriptor *instances,
                    instance_acceleration_structure accelerationStructure,
                    visible_function_table<IntersectionFunction> intersectionFunctionTable,
                    device AreaLight *areaLights,
                    thread PathVertex *cameraVertices,
                    int maxPathLength,
                    thread float3 &directLightingContribution,
                    device LightTriangle *lightTriangles,
                    array<texture2d<float>, MAX_TEXTURES> textureArray
                    )
{
    constant Camera& camera = uniforms.camera;
    
    thread PathVertex& cameraVertex = cameraVertices[0];
    cameraVertex.position = camera.position;
    cameraVertex.normal = camera.forward;
    cameraVertex.throughput = float3(1.0f, 1.0f, 1.0f);
    cameraVertex.forwardPDF = 1.0f;
    cameraVertex.reversePDF = 0.0f;
    cameraVertex.is_delta = true;
    cameraVertex.type = CAMERA_VERTEX;
    
    int pathLength = 1;
    float3 throughput = float3(1.0f);
    
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;

    ray ray;
    ray.origin = camera.position + camera.forward * 1e-4f;
    ray.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
    ray.max_distance = INFINITY;
            
    for (int bounce = 0; bounce < maxPathLength - 1; bounce++) {
        ray.direction = normalize(ray.direction);

        IntersectionResult intersection = intersect(ray,
                                                    bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_PRIMARY,
                                                    resources,
                                                    instances,
                                                    accelerationStructure,
                                                    intersectionFunctionTable,
                                                    false);
        
        if (intersection.type == intersection_type::none) {
            break;
        }
        
        unsigned int instanceIndex = intersection.instance_id;
        unsigned int mask = instances[instanceIndex].mask;
        float4x3 objectToWorldTransform = intersection.object_to_world_transform;
        float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;
        
        thread PathVertex& currVertex = cameraVertices[pathLength];
        currVertex.position = intersectionPoint;
        currVertex.throughput = throughput;
        currVertex.incoming_direction = -ray.direction;
        currVertex.type = CAMERA_VERTEX;
        
        unsigned primitiveIndex = intersection.primitive_id;
        unsigned int resourceIndex = instances[instanceIndex].accelerationStructureIndex;
        float2 barycentric_coords = intersection.triangle_barycentric_coord;
        
        if (mask & GEOMETRY_MASK_TRIANGLE) {
            device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
            
            float3 objectNormal = interpolateVertexAttribute(triangleResources.vertexNormals, primitiveIndex, barycentric_coords);
            float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldTransform));
            currVertex.normal = worldNormal;
        
            Material material = triangleResources.vertexMaterials[primitiveIndex];
            
            currVertex.material = material;
            
            float2 uv = interpolateVertexAttribute(triangleResources.vertexUVs, primitiveIndex, barycentric_coords);
            uv.y = 1 - uv.y;
            currVertex.material_color = interpolateVertexAttribute(triangleResources.vertexColors, primitiveIndex, barycentric_coords);
            
            constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);

            texture2d<float> texture = textureArray[material.texture_index];
            float3 textureColor = texture.sample(textureSampler, uv).xyz;
            
//            float3 textureColor = triangleResources.texture.sample(textureSampler, uv * 5.0f).xyz;
            currVertex.material_color *= textureColor;

            throughput *= currVertex.material_color;
        }
        
        if (++pathLength >= maxPathLength) {
            break;
        }
                
        float4 r = float4(
                          scrambledHalton(offset, 5 + bounce * 13 + 4, uniforms.frameIndex),
                          scrambledHalton(offset, 5 + bounce * 13 + 5, uniforms.frameIndex),
                          scrambledHalton(offset, 5 + bounce * 13 + 6, uniforms.frameIndex),
                          scrambledHalton(offset, 5 + bounce * 13 + 7, uniforms.frameIndex)
                          );
        
        LightSample sample = sampleAreaLight(areaLights, lightTriangles, uniforms, currVertex.position, currVertex.normal, r);
        
        struct ray shadowRay;
        shadowRay.origin = currVertex.position + currVertex.normal * 1e-4f;
        shadowRay.direction = sample.direction;
        shadowRay.max_distance = sample.distance - 1e-4f;
        
        intersection = intersect(shadowRay,
                                 RAY_MASK_SHADOW,
                                 resources,
                                 instances,
                                 accelerationStructure,
                                 intersectionFunctionTable,
                                 true);
        
        if (mask & GEOMETRY_MASK_LIGHT) {
            directLightingContribution += throughput * sample.emission;
            break;
        }
        
        if (intersection.type == intersection_type::none) {
            float posToLight = dot(currVertex.normal, sample.direction); // angle between position normal and light direction
            float surfaceToLight = dot(sample.normal, -sample.direction); // angle between light surface normal and light direction
            
            if (posToLight > 0.0f && surfaceToLight > 0.0f) {
                float3 bsdfValue = evaluateBSDF(currVertex, sample.direction);
                float distanceSq = max(1e-6f, sample.distance * sample.distance);
                float3 directLight = throughput * bsdfValue * posToLight * sample.emission / (sample.PDF * distanceSq);
                
                directLightingContribution += directLight;
            }
        }
        
        BounceInfo bounceInfo = calculateBounce(ray, intersectionPoint, currVertex.normal, currVertex.material, offset, bounce, uniforms.frameIndex);
        currVertex.is_delta = bounceInfo.is_delta;
        ray = bounceInfo.ray;
        
        calculateNewPDFs(currVertex, cameraVertices[pathLength - 1], ray.direction, bounceInfo.normal);
    }
    
    return pathLength;
}

int traceLightPath(float2 pixel,
                   constant Uniforms & uniforms,
                   unsigned int offset,
                   device void *resources,
                   device MTLAccelerationStructureInstanceDescriptor *instances,
                   instance_acceleration_structure accelerationStructure,
                   visible_function_table<IntersectionFunction> intersectionFunctionTable,
                   device AreaLight *areaLights,
                   thread PathVertex *lightVertices,
                   int maxPathLength,
                   device LightTriangle *lightTriangles,
                   array<texture2d<float>, MAX_TEXTURES> textureArray
                   )
{
    float4 r = float4(scrambledHalton(offset, 50, uniforms.frameIndex),
                      scrambledHalton(offset, 51, uniforms.frameIndex),
                      scrambledHalton(offset, 52, uniforms.frameIndex),
                      scrambledHalton(offset, 53, uniforms.frameIndex));
    
    float lightRandom = scrambledHalton(offset, 49, uniforms.frameIndex);
    uint lightIndex = min(uint(lightRandom * uniforms.lightCount), uniforms.lightCount - 1);
    
    LightSample sample = sampleTriangle(areaLights[lightIndex], lightTriangles, 1.0f / float(uniforms.lightCount), float3(0.0f), r.yzw);
    
    sample.areaLight = areaLights[lightIndex];
    AreaLight selectedLight = sample.areaLight;
    
    float2 randomDirection = float2(scrambledHalton(offset, 40, uniforms.frameIndex), scrambledHalton(offset, 41, uniforms.frameIndex));
    float3 localDirection = sampleCosineWeightedHemisphere(randomDirection);
    float3 worldDirection = alignHemisphereWithNormal(localDirection, sample.normal);
    
    float cosTheta = max(0.0f, dot(worldDirection, sample.normal));
    float directionPDF = cosTheta / M_PI_F;
    
    lightVertices[0].position = sample.position + sample.normal * 1e-4f;
    lightVertices[0].normal = sample.normal;
    lightVertices[0].throughput = sample.emission;
    lightVertices[0].material_color = sample.emission;
    lightVertices[0].forwardPDF = sample.PDF * directionPDF;
    lightVertices[0].reversePDF = 0.0f;
    lightVertices[0].type = LIGHT_VERTEX;
    lightVertices[0].is_delta = false;
    
    int pathLength = 1;
        
    float3 throughput = lightVertices[0].throughput;
    
    ray ray;
    ray.origin = lightVertices[0].position;
    ray.direction = worldDirection;
    ray.max_distance = INFINITY;
     
    for (int bounce = 0; bounce < maxPathLength - 1; bounce++) {
        IntersectionResult intersection = intersect(ray,
                                                    RAY_MASK_PRIMARY,
                                                    resources,
                                                    instances,
                                                    accelerationStructure,
                                                    intersectionFunctionTable,
                                                    false);
        
        if (intersection.type == intersection_type::none) {
            break;
        }
        
        unsigned int instanceIndex = intersection.instance_id;
        unsigned int mask = instances[instanceIndex].mask;
        
        if (mask & GEOMETRY_MASK_LIGHT) {
            break;
        }
        
        float4x3 objectToWorldTransform = intersection.object_to_world_transform;
        float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;
        
        thread PathVertex& currVertex = lightVertices[pathLength];
        currVertex.position = intersectionPoint;
        currVertex.throughput = throughput;
        currVertex.incoming_direction = -ray.direction;
        currVertex.type = LIGHT_VERTEX;
        
        unsigned int primitiveIndex = intersection.primitive_id;
        unsigned int resourceIndex = instances[instanceIndex].accelerationStructureIndex;
        float2 barycentric_coords = intersection.triangle_barycentric_coord;
        
        float2 r = float2(scrambledHalton(offset, 5 + bounce * 13 + 6, uniforms.frameIndex),
                          scrambledHalton(offset, 5 + bounce * 13 + 7, uniforms.frameIndex));

        if (mask & GEOMETRY_MASK_TRIANGLE) {
            device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
            
            float3 objectNormal = interpolateVertexAttribute(triangleResources.vertexNormals, primitiveIndex, barycentric_coords);
            float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldTransform));
            currVertex.normal = worldNormal;
            
            Material material = triangleResources.vertexMaterials[primitiveIndex];

            currVertex.material = material;

            float2 uv = interpolateVertexAttribute(triangleResources.vertexUVs, primitiveIndex, barycentric_coords);
            uv.y = 1 - uv.y;

            currVertex.material_color = interpolateVertexAttribute(triangleResources.vertexColors, primitiveIndex, barycentric_coords);

            constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);
            
            texture2d<float> texture = textureArray[material.texture_index];
            float3 textureColor = texture.sample(textureSampler, uv).xyz;
            
//            float3 textureColor = triangleResources.texture.sample(textureSampler, uv * 5.0f).xyz;
            currVertex.material_color *= textureColor;
            throughput *= currVertex.material_color;
        }
                
        if (++pathLength >= maxPathLength) {
            break;
        }
        
        BounceInfo bounceInfo = calculateBounce(ray, intersectionPoint, currVertex.normal, currVertex.material, offset, bounce, uniforms.frameIndex);
        ray = bounceInfo.ray;
        currVertex.is_delta = bounceInfo.is_delta;
        
        calculateNewPDFs(currVertex, lightVertices[pathLength - 1], ray.direction, bounceInfo.normal);
        
        if (!currVertex.is_delta) {
            float cosTheta = max(0.01f, dot(ray.direction, currVertex.normal));
            currVertex.throughput *= cosTheta;
        }

        throughput = currVertex.throughput;
    }
    
    return pathLength;
}



uint2 projectToScreen(float3 worldPos, constant Uniforms& uniforms)
{
    float3 toPoint = worldPos - uniforms.camera.position;
    float zCam = dot(toPoint, uniforms.camera.forward);
    if (zCam <= 0.0f)
        return uint2(UINT_MAX, UINT_MAX);

    float3 normalizedRight = normalize(uniforms.camera.right);
    float3 normalizedUp = normalize(uniforms.camera.up);

    float fieldOfView = CAMERA_FOV_ANGLE * (M_PI_F / 180.0f);
    float imagePlaneHeight = tan(fieldOfView / 2.0f);
    float imagePlaneWidth = imagePlaneHeight * float(uniforms.width) / float(uniforms.height);

    float xProj = dot(toPoint, normalizedRight) / (zCam * imagePlaneWidth);
    float yProj = dot(toPoint, normalizedUp) / (zCam * imagePlaneHeight);
    
    float2 uv;
    uv.x = 0.5f + 0.5f * xProj;
    uv.y = 0.5f + 0.5f * yProj;
    
    if (uv.x < 0.0f || uv.x > 1.0f ||
        uv.y < 0.0f || uv.y > 1.0f)
    {
        return uint2(UINT_MAX, UINT_MAX);
    }

    uint px = uint(uv.x * float(uniforms.width - 1));
    uint py = uint(uv.y * float(uniforms.height - 1));
    return uint2(px, py);
}

void splat(
    texture2d<float, access::read_write> splatTex,
    constant Uniforms& uniforms,
    uint2 pixelCoordinate,
    float3 color,
    device atomic_float* splatBuffer
)
{
    if (pixelCoordinate.x >= uniforms.width || pixelCoordinate.y >= uniforms.height)
        return;
    
    float3 scaledColor = color * 1.0f;
    int k = 0;
    
    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            uint2 splatCoord = uint2(pixelCoordinate.x + i, pixelCoordinate.y + j);
            
            if (splatCoord.x >= uniforms.width || splatCoord.y >= uniforms.height)
                continue;
            
            uint width = uniforms.width;
            uint pixelIndex = (splatCoord.y * width + splatCoord.x) * 3;
            
            float dist = length(float2(i, j));
            float weight = exp(-(dist * dist) / (2.0f * k * k + 1.0f));
            float3 contribution = weight * scaledColor;
            
            atomic_fetch_add_explicit(&splatBuffer[pixelIndex], contribution.r, memory_order_relaxed);
            atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 1], contribution.g, memory_order_relaxed);
            atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 2], contribution.b, memory_order_relaxed);
        }
    }
}

float3 connectPaths(
    thread PathVertex *cameraVertices,
    int cameraPathLength,
    thread PathVertex *lightVertices,
    int lightPathLength,
    device void *resources,
    device MTLAccelerationStructureInstanceDescriptor *instances,
    instance_acceleration_structure accelerationStructure,
    visible_function_table<IntersectionFunction> intersectionFunctionTable,
                    texture2d<float, access::read_write> splatTex,
    constant Uniforms& uniforms,
    unsigned int offset,
    device atomic_float* splatBuffer
) {
    float3 totalContribution = float3(0.0f);
    
    for (int c = 1; c < cameraPathLength; c++) {
        for (int l = 0; l < lightPathLength; l++) {
            float3 contribution = float3(1.0f, 1.0f, 1.0f);
        
            if (c == 0 && l == 0) continue;
            if (c == 0 && l == 1) continue;
            
            if (cameraVertices[c].is_delta || lightVertices[l].is_delta)
                continue;

            thread PathVertex &cameraVertex = cameraVertices[c];
            thread PathVertex &lightVertex = lightVertices[l];

            float3 connectionVector = lightVertex.position - cameraVertex.position;
            float connectionDistance = length(connectionVector);
            float3 connectionDirection = connectionVector / connectionDistance;
            
            float cosCamera = dot(connectionDirection, cameraVertex.normal);
            float cosLight = dot(-connectionDirection, lightVertex.normal);
                        
            if (cosCamera <= 0.0f || cosLight <= 0.0f)
                continue;
            
            ray shadowRay;
            shadowRay.origin = cameraVertex.position + cameraVertex.normal * 1e-4f;
            shadowRay.direction = normalize(connectionDirection);
            shadowRay.max_distance = connectionDistance - 1e-4f;
            
            IntersectionResult intersection = intersect(
                shadowRay,
                RAY_MASK_SHADOW,
                resources,
                instances,
                accelerationStructure,
                intersectionFunctionTable,
                true
            );
            
            if (intersection.type != intersection_type::none) {
                
                unsigned int mask = instances[intersection.instance_id].mask;

                unsigned primitiveIndex = intersection.primitive_id;
                unsigned int resourceIndex = instances[intersection.instance_id].accelerationStructureIndex;

                device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
                Material material = triangleResources.vertexMaterials[primitiveIndex];
                float2 barycentric_coords = intersection.triangle_barycentric_coord;

                float opacity = material.opacity;
                
                if (opacity >= 0.9f) {
                    continue;
                } else {
                    contribution *= (1.0f - opacity);
                }
            }
            
            float3 brdfCamera = evaluateBSDF(cameraVertex, connectionDirection);
            float3 brdfLight = evaluateBSDF(lightVertex, -connectionDirection);
            float geometricTerm = (cosCamera * cosLight) / (connectionDistance * connectionDistance);
            
            float pdfConversionFactor = 1.0f;
            if (lightVertex.type == LIGHT_VERTEX && lightVertex.forwardPDF > 0) {
                pdfConversionFactor = connectionDistance * connectionDistance / cosLight;
            }
            
            contribution *= cameraVertex.throughput * brdfCamera *
                           geometricTerm * brdfLight * lightVertex.throughput *
                           pdfConversionFactor;
            totalContribution += contribution;
        }
        
        
        for (int l = 1; l < lightPathLength; l++) {
            if (lightVertices[l].is_delta || lightVertices[l].material.opacity < 0.1f)
                continue;
            
            bool isCausticPath = false;
            
            int hitSpecular = 0;
            
            for (int v = 1; v <= l; v++) {
                if (lightVertices[v].is_delta || lightVertices[v].material.metallic > 0.5f ||
                    lightVertices[v].material.opacity < 0.9f) {
                    hitSpecular++;
                } else if (hitSpecular >= 1) { // 1 to show mirror, 2 for glass
                    isCausticPath = true;
                    break;
                }
            }
            
            if (!isCausticPath) continue;

            float3 contribution = float3(1.0f);
            
            thread PathVertex& cameraVertex = cameraVertices[0];
            
            float3 toCamera = cameraVertex.position - lightVertices[l].position;
            float distanceToCamera = length(toCamera);
            float3 toCameraDir = toCamera / distanceToCamera;
            
            float cosLight = dot(toCameraDir, lightVertices[l].normal);
            
            if (cosLight <= 0.0f)
                continue;
            
            ray shadowRay;
            shadowRay.origin = lightVertices[l].position + lightVertices[l].normal * 1e-4f;
            shadowRay.direction = toCameraDir;
            shadowRay.max_distance = distanceToCamera - 1e-4f;
            
            IntersectionResult intersection = intersect(
                shadowRay,
                RAY_MASK_SHADOW,
                resources,
                instances,
                accelerationStructure,
                intersectionFunctionTable,
                true
            );
            
            if (intersection.type != intersection_type::none) {
                unsigned int primitiveIndex = intersection.primitive_id;
                unsigned int resourceIndex = instances[intersection.instance_id].accelerationStructureIndex;

                device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
                Material material = triangleResources.vertexMaterials[primitiveIndex];
                float opacity = material.opacity;
                
                if (opacity >= 0.9f) {
                    continue;
                } else {
                    contribution *= (1.0f - opacity);
                }
            }
            
            float3 brdfLight = evaluateBSDF(lightVertices[l], toCameraDir);
            float geometricTerm = cosLight / (distanceToCamera * distanceToCamera);
            
            contribution *= lightVertices[l].throughput * geometricTerm * brdfLight;
            
            uint2 pixelCoord = projectToScreen(lightVertices[l].position, uniforms);
                
            splat(splatTex, uniforms, pixelCoord, contribution, splatBuffer);
        }
    
        
    }
    
    return totalContribution;
}

kernel void raytracingKernel(uint2 tid [[thread_position_in_grid]],
                             uint sampleIndex [[thread_index_in_threadgroup]],
                             constant Uniforms & uniforms,
                             texture2d<unsigned int> randomTex,
                             texture2d<float> prevTex,
                             texture2d<float, access::read_write> dstTex,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             device AreaLight *areaLights,
                             instance_acceleration_structure accelerationStructure,
                             visible_function_table<IntersectionFunction> intersectionFunctionTable,
                             device atomic_float* splatBuffer,
                             texture2d<float> prevSplat,
                             texture2d<float, access::read_write> splatTex,
                             texture2d<float, access::write> finalImage,
                             device LightTriangle *lightTriangles,
                             array<texture2d<float>, MAX_TEXTURES> textureArray [[texture(8)]]
                             )
{

    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;
    
    unsigned int offset = randomTex.read(tid).x;
    
    float2 pixel = (float2)tid;
    float2 r = float2(scrambledHalton(offset, 8, uniforms.frameIndex),
                      scrambledHalton(offset, 9, uniforms.frameIndex));
    pixel += r;
    
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;
    
    ray cameraRay;
    cameraRay.origin = uniforms.camera.position;
    cameraRay.direction = normalize(uv.x * uniforms.camera.right +
                                  uv.y * uniforms.camera.up +
                                  uniforms.camera.forward);
    cameraRay.max_distance = INFINITY;
    
    PathVertex cameraVertices[MAX_CAMERA_PATH_LENGTH];
    PathVertex lightVertices[MAX_LIGHT_PATH_LENGTH];
    
    float3 accumulatedColor = float3(0.0f);
    float3 directLightingContribution = float3(0.0f);
    
    float3 entryPoint = float3(0.0f);
    bool hasStoredEntryPoint = false;
    float3 beer = float3(1.0f);
    
    int cameraPathLength = traceCameraPath(
                                           pixel,
                                           uniforms,
                                           offset,
                                           resources,
                                           instances,
                                           accelerationStructure,
                                           intersectionFunctionTable,
                                           areaLights,
                                           cameraVertices,
                                           MAX_CAMERA_PATH_LENGTH,
                                           directLightingContribution,
                                           lightTriangles,
                                           textureArray
                                           );
    
    int lightPathLength = traceLightPath(
                                         pixel,
                                         uniforms,
                                         offset,
                                         resources,
                                         instances,
                                         accelerationStructure,
                                         intersectionFunctionTable,
                                         areaLights,
                                         lightVertices,
                                         MAX_LIGHT_PATH_LENGTH,
                                         lightTriangles,
                                         textureArray
                                         );
    
    float3 indirectLighting = float3(0.0f);

    if (cameraPathLength > 0 && lightPathLength > 0) {
        indirectLighting += connectPaths(
                                         cameraVertices,
                                         cameraPathLength,
                                         lightVertices,
                                         lightPathLength,
                                         resources,
                                         instances,
                                         accelerationStructure,
                                         intersectionFunctionTable,
                                         splatTex,
                                         uniforms,
                                         offset,
                                         splatBuffer
                                         );
    }
    
    float indirectWeight = 1.0f;
    float3 totalLighting = (1.0f * 1.0f * directLightingContribution + 1.0f * 3.0f * indirectLighting * indirectWeight);
    float3 totalSplat = splatTex.read(tid).xyz;
    
    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(tid).xyz;
        prevColor *= uniforms.frameIndex;
        totalLighting += prevColor;
        totalLighting /= (uniforms.frameIndex + 1);
        
        float3 previousSplat = prevSplat.read(tid).xyz;
        previousSplat *= uniforms.frameIndex;
        totalSplat += previousSplat;
        totalSplat /= (uniforms.frameIndex + 1);
    }
    
    dstTex.write(float4(totalLighting, 1.0f), tid);
    splatTex.write(float4(totalSplat, 1.0f), tid);
    finalImage.write(float4(1.0f * totalLighting + 1.0f * 20.0f * totalSplat, 1.0f), tid);
}

kernel void clearAtomicBuffer(device atomic_float* atomicBuffer [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]],
                              texture2d<float, access::write> outputTexture [[texture(0)]],
                              uint2 gridSize [[threads_per_grid]]) {
    
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height())
        return;

    uint width = outputTexture.get_width();
    uint index = gid.y * width + gid.x;
    
    if (index < width * outputTexture.get_height()) {
        uint bufferIndex = index;
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3 + 1], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&atomicBuffer[bufferIndex * 3 + 2], 0.0f, memory_order_relaxed);
    }
}

kernel void finalizeAtomicBuffer(device atomic_float* atomicBuffer [[buffer(0)]],
                                texture2d<float, access::write> currentSplatTexture [[texture(0)]],
                                 texture2d<float, access::read> previousSplatTexture [[texture(1)]],
                                constant Uniforms &uniforms [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= currentSplatTexture.get_width() || gid.y >= currentSplatTexture.get_height())
        return;
    
    uint width = currentSplatTexture.get_width();
    uint index = (gid.y * width + gid.x) * 3;
    
    float3 currentFrameContribution = float3(
        atomic_load_explicit(&atomicBuffer[index + 0], memory_order_relaxed),
        atomic_load_explicit(&atomicBuffer[index + 1], memory_order_relaxed),
        atomic_load_explicit(&atomicBuffer[index + 2], memory_order_relaxed)
    );
    
    float3 previousAccumulation = previousSplatTexture.read(gid).xyz;
    
    float3 newAccumulation;
    if (uniforms.frameIndex > 0) {
        previousAccumulation *= uniforms.frameIndex;
        newAccumulation = previousAccumulation + currentFrameContribution;
        newAccumulation /= (uniforms.frameIndex + 1);
    } else {
        newAccumulation = currentFrameContribution;
    }
    
    currentSplatTexture.write(float4(newAccumulation, 1.0), gid);
}

// sample shaders below

// Screen filling quad in normalized device coordinates.
constant float2 quadVertices[] =
{
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut
{
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions.
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]])
{
    float2 position = quadVertices[vid];

    CopyVertexOut out;

    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;

    return out;
}

// Simple fragment shader which copies a texture and applies a simple tonemapping function.
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);

    float3 color = tex.sample(sam, in.uv).xyz;

    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which the screen can display.
    color = color / (1.0f + color);

    return float4(color, 1.0f);
}
