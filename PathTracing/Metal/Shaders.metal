//
//  Shaders.metal
//  PathTracing
//

#include "Shaders.h"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
using namespace raytracing;

constant unsigned int resourcesStride  [[function_constant(0)]];
constant bool useIntersectionFunctions [[function_constant(1)]];

#define DEBUG(...) os_log_default.log_info(__VA_ARGS__)

constant unsigned int primes[] =
{
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
    811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
    1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373,
    1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657,
    1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
    1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129,
    2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267
};

void debug(float x) {
    os_log_default.log_info("%f", x);
}

void debug(float3 w) {
    os_log_default.log_info("mag : %f : float3(%f, %f, %f)", length(w), w.x, w.y, w.z);
}

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

typedef intersector<triangle_data, instancing, world_space_data>::result_type IntersectionResult;

IntersectionResult intersect(ray ray,
                             unsigned int mask,
                             device void *resources,
                             device MTLAccelerationStructureInstanceDescriptor *instances,
                             instance_acceleration_structure accelerationStructure,
                             visible_function_table<IntersectionFunction> intersectionFunctionTable,
                             bool accept_any_intersection)
{
    intersection_params params;
    
    intersection_query<triangle_data, instancing> i;
    
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);

    params.accept_any_intersection(accept_any_intersection); // get any, not just the closest

    i.reset(ray, accelerationStructure, mask, params);

    i.next();
    
    IntersectionResult intersection;
    
    intersection.type = i.get_committed_intersection_type();
    intersection.distance = i.get_committed_distance();
    intersection.primitive_id = i.get_committed_primitive_id();
    intersection.geometry_id = i.get_committed_geometry_id();
    intersection.triangle_barycentric_coord = i.get_committed_triangle_barycentric_coord();
    intersection.instance_id = i.get_committed_instance_id();
    intersection.object_to_world_transform = i.get_committed_object_to_world_transform();
    
    return intersection;
}

#pragma mark PDFs and BSDFs

inline float lambertianDiffusePDF(float3 w, float3 n) {
    return max(dot(w, n), 1e-6f) / M_PI_F;
}

inline float3 lambertianDiffuseBSDF(float3 albedo) {
    return albedo / M_PI_F;
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

inline float sameSignClamp(float x, float epsilon) {
    if (x < 0.0f) return min(x, -epsilon);
    else return max(x, epsilon);
}

inline void createOrthonormalBasis(float3 position, float3 n, thread float3& t, thread float3& b) {
    float sign = copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b_val = n.x * n.y * a;
    t = float3(1.0f + sign * n.x * n.x * a, sign * b_val, -sign * n.x);
    b = float3(b_val, sign + n.y * n.y * a, -n.y);
    t = normalize(t);
    b = normalize(b);
}

float lambda(float3 w, float3 n, float alpha) {
    float cosTheta = abs(dot(w, n));
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = 1 - cosTheta2;
    
    return 0.5f * (sqrt(1.0f + alpha * alpha * sinTheta2 / cosTheta2) - 1.0f);
}

float G_Smith(float3 wi, float3 wo, float3 n, float alpha) {
    float lambda_wi = lambda(wi, n, alpha);
    float lambda_wo = lambda(wo, n, alpha);
    return 1.0f / (1.0f + lambda_wi + lambda_wo);
}

float G1_Smith(float3 w, float3 n, float alpha) {
    return 1.0f / (1.0f + lambda(w, n, alpha));
}

inline float D_GGX(float cosTH, float cosBH, float cosNH, float alpha_x, float alpha_y) {
    float x = cosTH / alpha_x;
    float y = cosBH / alpha_y;
    float z = cosNH;

    float denom = x * x + y * y + z * z;
    
    return 1.0f / (M_PI_F * alpha_x * alpha_y * denom * denom);
}

inline float interpolateAlpha(float cosTH, float cosBH, float alpha_x, float alpha_y) {
    float num = (cosTH * cosTH) * (alpha_x * alpha_x) + (cosBH * cosBH) * (alpha_y * alpha_y);
    float denom = (cosTH * cosTH) + (cosBH * cosBH);
    
    return sqrt(num / denom);
}

inline float calculateLuminance(float3 w) {
    return dot(w, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float calculateEpsilon(float3 position) {
    return min(1e-4f * length(position), 1e-6f);
}

inline float3 calculateOffset(float3 wo, float3 n, float epsilon) {
    return wo * 0.1f * epsilon + n * epsilon;
}
 
inline float calculateEta(PathVertex previous, PathVertex current, bool entering) {
    if (entering) return 1.0f / current.material.refraction;
    else return current.material.refraction;
}

inline float3 calculateConnectionDirection(PathVertex start, PathVertex end) {
    return normalize(end.position - start.position);
}

inline void calculateInteractionWeights(PathVertex vx, float3 fresnel, float eta, bool TIR, thread float &diffuseWeight, thread float &specularWeight, thread float &transmissionWeight) {
    Material material = vx.material;

    specularWeight = calculateLuminance(fresnel);
    transmissionWeight = (1.0f - material.opacity) * eta * eta * calculateLuminance(1.0f - fresnel);
    diffuseWeight = material.opacity * (1.0f - material.metallic) * calculateLuminance(vx.material_color);

    if (TIR) {
        specularWeight += transmissionWeight;
        transmissionWeight = 0.0f;
    }

    float totalWeight = specularWeight + transmissionWeight + diffuseWeight;

    if (totalWeight > 0.0f) {
        specularWeight /= totalWeight;
        transmissionWeight /= totalWeight;
        diffuseWeight /= totalWeight;
    }
}


PathVertex sampleTriangle(thread AreaLight areaLight,
                      device LightTriangle *lightTriangles,
                      float selectionPDF,
                      float3 r
                      )
{
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
    float3 position = u * triangle.v0 + v * triangle.v1 + w * triangle.v2;
    float epsilon = calculateEpsilon(position);

    PathVertex vx;
    vx.normal = normalize(cross(edge1, edge2));
    vx.position = position + calculateOffset(vx.normal, vx.normal, epsilon);
    vx.material_color = u * triangle.emission0 + v * triangle.emission1 + w * triangle.emission2;
    vx.forwardPDF = selectionPDF / areaLight.totalArea;
    vx.throughput = vx.material_color / vx.forwardPDF;
    vx.reversePDF = 0.0f;
    vx.is_delta = false;
    vx.in_medium = false;
    vx.mediumDistance = 0.0f;
    vx.type = LIGHT_VERTEX;

    return vx;
}

PathVertex sampleAreaLight(device AreaLight *areaLights,
                       device LightTriangle *lightTriangles,
                       constant Uniforms& uniforms,
                       float4 r
                       )
{
    PathVertex vx;
    float weights[MAX_AREA_LIGHTS];
    float totalWeight = 0.0f;
    
    for (unsigned int i = 0; i < uniforms.lightCount; i++) {
        AreaLight light = areaLights[i];
        float power = calculateLuminance(light.color) * light.totalArea;
                        
        weights[i] = power;
        totalWeight += weights[i];
    }
    
    unsigned int idx = 0;
    float lightPDF = 1.0f;
    
    if (totalWeight < 1e-4f) {
        idx = r.x * float(uniforms.lightCount);
        lightPDF = 1.0f / float(uniforms.lightCount);
    } else {
        float random = r.x * totalWeight;
        float accumWeight = 0.0f;
        
        for (unsigned int i = 0; i < uniforms.lightCount; i++) {
            accumWeight += weights[i];
            
            if (accumWeight >= random) {
                idx = i;
                lightPDF = weights[i] / totalWeight;
                break;
            }
        }
    }
        
    vx = sampleTriangle(areaLights[idx], lightTriangles, lightPDF, r.yzw);
    return vx;
}

float convertDensity(float PDF, PathVertex vx, PathVertex nxt) {
    float3 w = nxt.position - vx.position;
    float d2_inv = 1.0f / length_squared(w);
    float cosN = abs(dot(nxt.normal, w * sqrt(d2_inv)));
    return PDF * cosN * d2_inv;
}

float evaluateLightPDF(PathVertex vx, PathVertex nxt) {
    float3 wo = calculateConnectionDirection(vx, nxt);
    float d2_inv = 1.0f / length_squared(wo);
    wo *= sqrt(d2_inv);
    float PDF;
    
    PDF = abs(dot(vx.normal, wo)) / M_PI_F; // PDF of sampling nxt from vx
    float cosN = abs(dot(nxt.normal, wo)); // convert to area measure

    return PDF * d2_inv * cosN;
}

float cameraPDF(PathVertex vx, PathVertex nxt) {
    float3 wo = calculateConnectionDirection(vx, nxt);
    return 1.0f / (A * pow(dot(wo, vx.normal), 3.0f));
}

float evaluatePDF(thread PathVertex &vx, thread PathVertex &prev, thread PathVertex &nxt) {
    if (vx.type & LIGHT_VERTEX) {
        return evaluateLightPDF(vx, nxt);
    }
    if (vx.type & CAMERA_VERTEX) {
        return convertDensity(cameraPDF(vx, nxt), vx, nxt); // pinhole
    }
    
    float3 wi = -calculateConnectionDirection(prev, vx);
    float3 wo = calculateConnectionDirection(vx, nxt);
        
    float3 n = vx.normal;
    Material material = vx.material;
    
    float dielectricF0 = pow((material.refraction - 1.0f) / (material.refraction + 1.0f), 2.0f);
    float3 F0 = mix(float3(dielectricF0), vx.material_color, material.metallic);
    float cosIN = dot(wi, n);
    bool inside = false;
    
    if (cosIN < 0.0f) {
        n = -n;
        inside = true;
    }
    
    float3 T, B;
//    createOrthonormalBasis(n, T, B);
    createOrthonormalBasis(vx.position, n, T, B);
    
    cosIN = max(abs(cosIN), 1e-4f);
    float cosON = sameSignClamp(dot(wo, n), 1e-4f);
    float3 fresnel = F0 + (float3(1.0f) - F0) * pow(1.0f - cosIN, 5.0f);
    float eta = !inside ? (1.0f / material.refraction) : material.refraction;
    
    float diffuseWeight, specularWeight, transmissionWeight;
    bool TIR = (material.opacity < 1.0f) && (1.0 - eta * eta * (1.0 - cosIN * cosIN) < 0.0);
    calculateInteractionWeights(vx, fresnel, eta, TIR, diffuseWeight, specularWeight, transmissionWeight);

    float totalPDF = 0.0f;
    
    if (cosON > 0.0f) {
        
        totalPDF += diffuseWeight * (cosON / M_PI_F);
        
        float3 H = normalize(wi + wo);
        if (dot(H, n) < 0.0f) H = -H;
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);
        float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
        float G1_wi = G1_Smith(wi, n, alpha);
//        float G = G_Smith(wi, wo, n, alpha);
//        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIN, 5.0f);
        
        totalPDF += specularWeight * (D * G1_wi) / (4.0f * cosIN);
    }
    else {
        float eta = inside ? material.refraction : (1.0f / material.refraction);
        float3 H = normalize(eta * wi + wo);
        
        float cosIH = max(abs(dot(wi, H)), 1e-4f);
        float cosOH = min(dot(wo, H), -1e-4f);
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);
        float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
        float G1_wi = G1_Smith(wi, n, alpha);
//        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIN, 5.0f);
        
        float denomF = eta * cosIH + cosOH;
        float jacobianF = (eta * eta * abs(cosOH)) / (denomF * denomF + 1e-10f);
        totalPDF += transmissionWeight * (D * G1_wi * abs(cosIH) / abs(cosIN)) * jacobianF;
    }
    
    return convertDensity(totalPDF, vx, nxt);
}

float calculateMISWeight(thread PathVertex *cameraVertices, thread PathVertex *lightVertices,
                         int c, int l,
                         PathVertex sampled
                         )
{
    if (c + l == 2) return 1.0f;
    auto remap0 = [&](float x) -> float { return x != 0.0f ? x : 1.0f; };

    int ci = c - 1;
    int cip = ci - 1;
    int li = l - 1;
    int lip = li - 1;

    PathVertex origVx;
    
    if (l == 1) {
        origVx = lightVertices[0];
        lightVertices[0] = sampled;
    }
    
    float originalCameraReverse = cameraVertices[ci].reversePDF;
    bool  origCamDelta   = cameraVertices[ci].is_delta;
    float origCamPrevRev = (cip >= 0) ? cameraVertices[cip].reversePDF : 0.0f;
    bool  origCamPrevDel = (cip >= 0) ? cameraVertices[cip].is_delta   : false;

    float origLgtRev     = lightVertices[li].reversePDF;
    bool  origLgtDelta   = lightVertices[li].is_delta;
    float origLgtPrevRev = (lip >= 0) ? lightVertices[lip].reversePDF : 0.0f;
    bool  origLgtPrevDel = (lip >= 0) ? lightVertices[lip].is_delta   : false;
                
    if (ci >= 0) {
        if (li >= 0) {
            cameraVertices[ci].reversePDF = evaluatePDF(lightVertices[li], lightVertices[lip], cameraVertices[ci]);
        } else {
            cameraVertices[ci].reversePDF = lightVertices[0].forwardPDF; // FIXME: may not work for multiple lights
        }
        cameraVertices[ci].is_delta = false;
    }
    
    if (cip >= 0) {
        if (li >= 0) {
            cameraVertices[cip].reversePDF = evaluatePDF(cameraVertices[ci], lightVertices[li], cameraVertices[cip]);
        } else {
            cameraVertices[cip].reversePDF = evaluateLightPDF(cameraVertices[ci], cameraVertices[cip]);
        }
    }
    
    if (li >= 0) {
        lightVertices[li].reversePDF = evaluatePDF(cameraVertices[ci], cameraVertices[cip], lightVertices[li]);
        lightVertices[li].is_delta = false;
    }
    
    if (lip >= 0) {
        lightVertices[lip].reversePDF = evaluatePDF(lightVertices[li], cameraVertices[ci], lightVertices[lip]);

    }
    
    float sum = 0.0f;
    float r = 1.0f;

    for (int i = ci; i > 0; --i) {
        r *= remap0(cameraVertices[i].reversePDF) / remap0(cameraVertices[i].forwardPDF);

        if (!cameraVertices[i].is_delta && !cameraVertices[i - 1].is_delta)
            sum += r;
    }

    r = 1;
    
    for (int i = li; i >= 0; --i) {
        r *= remap0(lightVertices[i].reversePDF) / remap0(lightVertices[i].forwardPDF);
        bool prevDelta = (i > 0) ? lightVertices[i - 1].is_delta : false;

        if (!lightVertices[i].is_delta && !prevDelta)
            sum += r;
    }
        
    cameraVertices[ci].reversePDF = originalCameraReverse;
    cameraVertices[ci].is_delta   = origCamDelta;
    if (cip >= 0) {
        cameraVertices[cip].reversePDF = origCamPrevRev;
        cameraVertices[cip].is_delta   = origCamPrevDel;
    }

    lightVertices[li].reversePDF = origLgtRev;
    lightVertices[li].is_delta = origLgtDelta;
    if (lip >= 0) {
        lightVertices[lip].reversePDF = origLgtPrevRev;
        lightVertices[lip].is_delta   = origLgtPrevDel;
    }
    
    if (l == 1) {
        lightVertices[0] = origVx;
    }

    return 1.0f / (1.0f + sum);
}

bool isVisible(float3 position, float3 n, float3 direction, float connectionDistance,
               device void *resources,
               device MTLAccelerationStructureInstanceDescriptor *instances,
               instance_acceleration_structure accelerationStructure,
               visible_function_table<IntersectionFunction> intersectionFunctionTable
               )
{
    ray shadowRay;
    float epsilon = calculateEpsilon(position);
    
    shadowRay.direction = direction;
    shadowRay.origin = position + calculateOffset(shadowRay.direction, n, epsilon);
    shadowRay.min_distance = epsilon;
    shadowRay.max_distance = connectionDistance - epsilon;
    
    IntersectionResult intersection = intersect(
                                                shadowRay,
                                                RAY_MASK_SHADOW,
                                                resources,
                                                instances,
                                                accelerationStructure,
                                                intersectionFunctionTable,
                                                true
                                                );
    
    return intersection.type == intersection_type::none;
}

float3 evaluateBSDF(thread PathVertex &vx, thread PathVertex &nxt) {
    float3 wi = vx.incoming_direction;
    float3 wo = calculateConnectionDirection(vx, nxt);
    float3 n = vx.normal;
    Material material = vx.material;
    
    float dielectricF0 = pow((material.refraction - 1.0f) / (material.refraction + 1.0f), 2.0f);
    float3 F0 = mix(float3(dielectricF0), vx.material_color, material.metallic);
    float cosIN = dot(wi, n);
    bool inside = false;
    
    if (cosIN < 0.0f) {
        n = -n;
        inside = true;
    }
    
    float3 T, B;
//    createOrthonormalBasis(n, T, B);
    createOrthonormalBasis(vx.position, n, T, B);
    
    cosIN = max(abs(cosIN), 1e-4f);
    float cosON = sameSignClamp(dot(wo, n), 1e-4f);
    float3 fresnel = F0 + (float3(1.0f) - F0) * pow(1.0f - cosIN, 5.0f);
    float eta = !inside ? (1.0f / material.refraction) : material.refraction;
    
    float diffuseWeight, specularWeight, transmissionWeight;
    bool TIR = (material.opacity < 1.0f) && (1.0 - eta * eta * (1.0 - cosIN * cosIN) < 0.0);
    calculateInteractionWeights(vx, fresnel, eta, TIR, diffuseWeight, specularWeight, transmissionWeight);
    
    float3 totalBSDF = float3(0.0f);
    
    if (cosIN * cosON > 0.0f) { // reflection
        
        totalBSDF += diffuseWeight * (vx.material_color / M_PI_F);
        
        float3 H = normalize(wi + wo);
        if (dot(H, n) < 0.0f) H = -H;
        
        float cosIH = max(dot(wi, H), 1e-4f);
        float cosOH = max(dot(wo, H), 1e-4f);
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);
        float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
        float G1_wi = G1_Smith(wi, n, alpha);
        float G = G_Smith(wi, wo, n, alpha);
        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIH, 5.0f);
        if (TIR) fresnel = float3(1.0f);
        
        totalBSDF += specularWeight * (D * G * fresnel) / (4.0f * cosIN * cosON);
    }
    else { // transmission
        float eta = inside ? material.refraction : (1.0f / material.refraction);
        float3 H = normalize(eta * wi + wo);
        
        float cosIH = max(dot(wi, H), 1e-4f);
        float cosOH = max(dot(wo, H), 1e-4f);
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);
        float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
        float G1_wi = G1_Smith(wi, n, alpha);
        float G = G_Smith(wi, wo, n, alpha);
        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIH, 5.0f);
        
        float denomF = eta * cosIH + cosOH;
        float jacobianF = (eta * eta * abs(cosOH)) / (denomF * denomF + 1e-10f);
                
        float denomBSDF = eta * cosIH + cosOH;
        float jacobian = eta * eta * abs(cosIH * cosOH) / (denomBSDF * denomBSDF + 1e-10f);
        float3 BSDF = transmissionWeight * D * G * (1.0f - fresnel) * jacobian / abs(cosIN * cosON + 1e-10f);
        
        if (vx.in_medium || nxt.in_medium) { // connection from outside to inside
            float connectionDistance = length(vx.position - nxt.position);
            BSDF *= exp(-material.absorption * connectionDistance);
        }
        
        totalBSDF += transmissionWeight * BSDF;
    }
    
    return totalBSDF;
}

struct TriangleResources
{
    device float3 *vertexNormals;
    device float3 *vertexColors;
    device Material *vertexMaterials;
    device float2 *vertexUVs;
};

float2 concentricSampleDisk(float2 u) {
    float2 u_offset = 2.0f * u - 1.0f;
    if (u_offset.x == 0.0f && u_offset.y == 0.0f) return float2(0, 0);

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

float3 sampleGGXNormal(float3 wi, float alpha_x, float alpha_y, float2 r) {
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
    float3 H = normalize(float3(N.x * alpha_x, N.y * alpha_y, max(-10.0, N.z)));
    
    return H;
}

float3 diffuse(thread PathVertex& currVertex, thread PathVertex& prevVertex, thread float& forwardPDF, thread float& reversePDF, float3 wi, float2 r) {
    float3 n = currVertex.normal;
    
    float3 diffuseDir = normalize(sampleCosineWeightedHemisphere(r));
    float3 wo = normalize(alignHemisphereWithNormal(diffuseDir, n));
    
    forwardPDF = lambertianDiffusePDF(wo, n);
    reversePDF = lambertianDiffusePDF(wi, n);
    currVertex.BSDF = lambertianDiffuseBSDF(currVertex.material_color);
    currVertex.is_delta = false;
        
    float epsilon = calculateEpsilon(currVertex.position);
    currVertex.position += calculateOffset(wo, n, epsilon);
    return wo;
}

float3 specular(thread PathVertex& vx, thread PathVertex& prev, thread float& forwardPDF, thread float& reversePDF, float3 wi, float2 r, bool TIR) {
    float3 n = vx.normal;
    Material material = vx.material;

    if (dot(wi, n) < 0.0f) {
        n = -n;
    }

    float cosIN = max(dot(wi, n), 1e-4f);

    float3 wo;
    
    float dielectricF0 = pow((material.refraction - 1.0f) / (material.refraction + 1.0f), 2.0f);
    float3 F0 = mix(float3(dielectricF0), vx.material_color, material.metallic);

    if (max(material.roughness_x, material.roughness_y) < 0.01f) {
        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIN, 5.0f);
        if (TIR) fresnel = float3(1.0f);

        vx.is_delta = true;
        forwardPDF = 0.0f;
        reversePDF = 0.0f;
        vx.BSDF = fresnel / cosIN; // cancel out the cosine in the throughput update; helps against dark edges (cosIN = cosON)

        wo = reflect(-wi, n);
        
    } else {
        
        float3 T, B;
        createOrthonormalBasis(vx.position, n, T, B);
                
        float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
                
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);
        
        float3 localH = sampleGGXNormal(wiLocal, alpha_x, alpha_y, r);
        float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);
        
        wo = reflect(-wi, H);
        
        if (dot(wo, n) < 0.0f || dot(n, H) < 0.0f) { // needed
            forwardPDF = 0.0f;
            reversePDF = 0.0f;
            vx.BSDF = float3(0.0f, 0.0f, 0.0f);
            
        } else {

            float cosON = max(dot(wo, n), 1e-4f);
            float cosIH = max(dot(wi, H), 1e-4f);
            float cosNH = max(dot(n, H), 1e-4f);
            float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
            float G1_wi = G1_Smith(wi, n, alpha);
            float G1_wo = G1_Smith(wo, n, alpha);
            
            forwardPDF = (D * G1_wi) / (4.0f * cosIN);
            reversePDF = (D * G1_wo) / (4.0f * cosON);
            
            float G = G_Smith(wi, wo, n, alpha);
            float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIH, 5.0f);
            if (TIR) fresnel = float3(1.0f);

            vx.BSDF = (D * G * fresnel) / (4.0f * cosIN * cosON);
            
            vx.is_delta = false;
        }
    }
    
    if (material.opacity > 0.0f && material.metallic < 1.0f) vx.BSDF *= cosIN;
    
    float epsilon = calculateEpsilon(vx.position);
    vx.position += calculateOffset(wo, n, epsilon);
    return wo;
}


float3 transmission(thread PathVertex& currVertex, thread PathVertex& prevVertex, thread float& forwardPDF, thread float& reversePDF, float3 wi, float2 r, float eta) {
    float3 n = currVertex.normal;
    Material material = currVertex.material;
    
    bool entering = dot(wi, n) > 0.0f;
//    float eta = entering ? (1.0f / material.refraction) : material.refraction;
    n = entering ? n : -n;
    
    float3 wo;
    
    float3 F0 = float3(pow((material.refraction - 1.0f) / (material.refraction + 1.0f), 2.0f)); // assume metals cant be transmissive
    float cosIN = max(dot(wi, n), 1e-10f);
            
    float3 absorption = material.absorption;

    if (max(material.roughness_x, material.roughness_y) < 0.01f) {
        wo = refract(-wi, n, eta);
        float cosON = abs(dot(wo, n));
        
        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIN, 5.0f);
        currVertex.BSDF = (1.0f - fresnel) * eta * eta / cosON;
            
        if (entering) {
            currVertex.in_medium = true;
        } else {
            currVertex.in_medium = false;
            float3 attenuation = exp(-absorption * currVertex.mediumDistance);
            currVertex.BSDF *= attenuation;
            currVertex.mediumDistance = 0.0f;
        }
        
        forwardPDF = 0.0f;
        reversePDF = 0.0f;
        currVertex.is_delta = true;

    } else {
        float3 T, B;
        createOrthonormalBasis(currVertex.position, n, T, B);
        
        float3 wiLocal = normalize(float3(dot(wi, T), dot(wi, B), dot(wi, n)));
                
        float alpha_x = max(material.roughness_x * material.roughness_x, 0.0101f);
        float alpha_y = max(material.roughness_y * material.roughness_y, 0.0101f);

        float3 localH = sampleGGXNormal(wiLocal, alpha_x, alpha_y, r);
        float3 H = normalize(T * localH.x + B * localH.y + n * localH.z);
        
        float cosTH = sameSignClamp(dot(T, H), 1e-4f);
        float cosBH = sameSignClamp(dot(B, H), 1e-4f);
        float cosNH = sameSignClamp(dot(n, H), 1e-4f);
        float alpha = interpolateAlpha(cosTH, cosBH, alpha_x, alpha_y);

        if (dot(n, H) < 0.0f) {
            H = -H;
        }

        wo = refract(-wi, H, eta);

        if (dot(wo, n) > 0.0f) {
            forwardPDF = 0.0f;
            reversePDF = 0.0f;
            currVertex.BSDF = float3(0.0f, 0.0f, 0.0f);
            return wo;
        }
                    
        float cosON = min(dot(wo, n), -1e-4f);
        float cosIH = max(dot(wi, H), 1e-4f);
        float cosOH = min(dot(wo, H), -1e-4f);

        float D = D_GGX(cosTH, cosBH, cosNH, alpha_x, alpha_y);
        float G1_wi = G1_Smith(wi, n, alpha);
        float G1_wo = G1_Smith(wo, n, alpha);
        float eta_inv = 1.0f / eta;

        float denomF = eta * cosIH + cosOH;
        float jacobianF = (eta * eta * abs(cosOH)) / (denomF * denomF + 1e-10f);
        forwardPDF = (D * G1_wi * abs(cosIH) / abs(cosIN)) * jacobianF;
        
        float denomW = eta_inv * cosOH + cosIH;
        float jacobianW = (eta_inv * eta_inv * abs(cosIH)) / (denomW * denomW + 1e-10f);
        reversePDF = (D * G1_wo * abs(cosOH) / abs(cosON)) * jacobianW;
        
        float G = G_Smith(wi, wo, n, alpha);
        float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosIH, 5.0f);
        
        float denomBSDF = eta * cosIH + cosOH;
        float jacobian = eta * eta * abs(cosIH * cosOH) / (denomBSDF * denomBSDF + 1e-10f);
        currVertex.BSDF = D * G * (1.0f - fresnel) * jacobian / abs(cosIN * cosON + 1e-10f);
        
        if (entering) {
            currVertex.in_medium = true;
        } else {
            currVertex.in_medium = false;
            float3 attenuation = exp(-absorption * currVertex.mediumDistance);
            currVertex.BSDF *= attenuation;
            currVertex.mediumDistance = 0.0f;
        }

        currVertex.is_delta = false;
    }
    

    float epsilon = calculateEpsilon(currVertex.position);
    currVertex.position += wo * epsilon;
    return wo;
}

float3 calculateBounce(thread PathVertex& currVertex, thread PathVertex& prevVertex, thread float& forwardPDF, float3 wi, float2 r2, float r1, float r0) {
    Material material = currVertex.material;
    wi = -wi; // convert to pointing away from surface
    float3 n = currVertex.normal;
    bool entering = dot(wi, currVertex.normal) > 0.0f;
    if (entering) n = -n;
    float eta = calculateEta(prevVertex, currVertex, entering); // FIXME: fix this
    
    float3 wo;
    
    float dielectricF0 = pow((material.refraction - 1.0f) / (material.refraction + 1.0f), 2.0f);
    float3 F0 = mix(float3(dielectricF0), currVertex.material_color, material.metallic);
    float cosIN = abs(dot(wi, n));
    float3 fresnel = F0 + (float3(1.0f) - F0) * pow(1.0f - cosIN, 5.0f);
    bool TIR = (material.opacity < 1.0f) && (1.0 - eta * eta * (1.0 - cosIN * cosIN) < 0.0);
    float diffuseWeight, specularWeight, transmissionWeight;

    calculateInteractionWeights(currVertex, fresnel, eta, TIR, diffuseWeight, specularWeight, transmissionWeight);
    float reversePDF, scaledPDF;

    if (r1 < diffuseWeight) {
        wo = diffuse(currVertex, prevVertex, forwardPDF, reversePDF, wi, r2);

        scaledPDF = (forwardPDF != 0.0f ? forwardPDF : 1.0f) * diffuseWeight;
        forwardPDF *= diffuseWeight;
        reversePDF *= diffuseWeight;
    }
    else if (TIR || r1 < diffuseWeight + specularWeight) {
        wo = specular(currVertex, prevVertex, forwardPDF, reversePDF, wi, r2, TIR);
        
        scaledPDF = (forwardPDF != 0.0f ? forwardPDF : 1.0f) * specularWeight;
        forwardPDF *= specularWeight;
        reversePDF *= specularWeight;
    }
    else {
        wo = transmission(currVertex, prevVertex, forwardPDF, reversePDF, wi, r2, eta);
        
        scaledPDF = (forwardPDF != 0.0f ? forwardPDF : 1.0f) * transmissionWeight;
        forwardPDF *= transmissionWeight;
        reversePDF *= transmissionWeight;
    }
        
    currVertex.throughput *= currVertex.BSDF * abs(dot(wo, n)) / scaledPDF;
#ifdef BDPT
    prevVertex.reversePDF = convertDensity(reversePDF, currVertex, prevVertex);
#endif
    return wo;
}

int tracePath(thread ray &ray,
              float3 throughput,
              int type,
              thread float& forwardPDF,
              float2 pixel,
              constant Uniforms & uniforms,
              unsigned int offset,
              device void *resources,
              device MTLAccelerationStructureInstanceDescriptor *instances,
              instance_acceleration_structure accelerationStructure,
              visible_function_table<IntersectionFunction> intersectionFunctionTable,
              device AreaLight *areaLights,
              thread PathVertex *vertices,
              int maxPathLength,
              thread float3 &directLightingContribution,
              device LightTriangle *lightTriangles,
              array<texture2d<float>, MAX_TEXTURES> textureArray,
              thread bool &hitLight
              )
{
    int pathLength = 1;
        
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
            if (type == CAMERA_VERTEX) directLightingContribution += float3(5.0f) * throughput;
            break;
        }
        
        unsigned int instanceIndex = intersection.instance_id;
        unsigned int mask = instances[instanceIndex].mask;
        float4x3 objectToWorldTransform = intersection.object_to_world_transform;
        float3 intersectionPoint = ray.origin + ray.direction * intersection.distance;
        
        thread PathVertex& currVertex = vertices[pathLength];
        thread PathVertex& prevVertex = vertices[pathLength - 1];
        currVertex.position = intersectionPoint;
        currVertex.throughput = throughput;
        currVertex.reversePDF = 0.0f;
        currVertex.incoming_direction = -ray.direction;
        currVertex.mediumDistance = prevVertex.mediumDistance;
        currVertex.mediumDistance += prevVertex.in_medium ? length(currVertex.position - prevVertex.position) : 0.0f;
        currVertex.type = SURFACE_VERTEX;
        
        unsigned primitiveIndex = intersection.primitive_id;
        unsigned int resourceIndex = instances[instanceIndex].accelerationStructureIndex;
        float2 barycentric_coords = intersection.triangle_barycentric_coord;
        
        if (true) {
            device TriangleResources& triangleResources = *(device TriangleResources *)((device char *)resources + resourcesStride * resourceIndex);
            
            float3 objectNormal = interpolateVertexAttribute(triangleResources.vertexNormals, primitiveIndex, barycentric_coords);
            float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldTransform));
            currVertex.normal = worldNormal;
            
            Material material = triangleResources.vertexMaterials[primitiveIndex];
            
            currVertex.material = material;
            
            float2 uv = interpolateVertexAttribute(triangleResources.vertexUVs, primitiveIndex, barycentric_coords);
            uv.y = 1 - uv.y;
//            uv.x = 1 - uv.x;

            currVertex.material_color = interpolateVertexAttribute(triangleResources.vertexColors, primitiveIndex, barycentric_coords);
            
            constexpr sampler textureSampler(min_filter::linear, mag_filter::linear, mip_filter::none, s_address::repeat, t_address::repeat);

            if (material.texture_index != -1) {
                texture2d<float> texture = textureArray[material.texture_index];
                float4 textureValue = texture.sample(textureSampler, uv);
                float3 textureColor = textureValue.w > 0.0f ? textureValue.xyz : 1.0f;
                
                currVertex.material_color *= textureColor;
            }
        }
        
        currVertex.forwardPDF = convertDensity(forwardPDF, currVertex, prevVertex);
                
        if (mask & GEOMETRY_MASK_LIGHT) {
            if (type == CAMERA_VERTEX) {
#ifdef BDPT
                if (bounce == 0)
#endif
                    directLightingContribution += float3(50.0f) * throughput;
                
                hitLight = true;
            }
            break;
        }

        if (++pathLength >= maxPathLength) {
            break;
        }
                                                            
        float2 r2 = float2(scrambledHalton(offset, 5 + bounce * 13 + 16, uniforms.frameIndex), scrambledHalton(offset, 5 + bounce * 13 + 15, uniforms.frameIndex));
        float r1 = scrambledHalton(offset, 5 + bounce * 13 + 17, uniforms.frameIndex);
        float r0 = scrambledHalton(offset, 5 + bounce * 13 + 18, uniforms.frameIndex);
        
        ray.direction = calculateBounce(currVertex, prevVertex, forwardPDF, ray.direction, r2, r1, r0);
        ray.min_distance = calculateEpsilon(currVertex.position);
        ray.origin = currVertex.position;
        
        throughput = currVertex.throughput;
                
        if (bounce > 3) {
            float throughputLuminance = calculateLuminance(throughput);
            float P_surv = min(0.99f, throughputLuminance);

            float rr = scrambledHalton(offset, 5 + bounce * 13 + 40, uniforms.frameIndex);

            if (rr > P_surv) {
                break;
            }

            throughput /= P_surv;
        }
    }

    return pathLength;
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
                    array<texture2d<float>, MAX_TEXTURES> textureArray,
                    thread bool &hitLight
                    )
{
    constant Camera& camera = uniforms.camera;
    
    thread PathVertex& cameraVertex = cameraVertices[0];
    cameraVertex.position = camera.position;
    cameraVertex.normal = camera.forward;
    cameraVertex.throughput = float3(1.0f);
    cameraVertex.forwardPDF = 0.0f;
    cameraVertex.reversePDF = 0.0f;
    cameraVertex.mediumDistance = 0.0f;
    cameraVertex.in_medium = false;
    cameraVertex.is_delta = true;
    cameraVertex.type = CAMERA_VERTEX;
        
    float2 uv = pixel / float2(uniforms.width, uniforms.height);
    uv = uv * 2.0f - 1.0f;

    ray ray;
    ray.origin = camera.position + camera.forward * 1e-4f;
    ray.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
    ray.min_distance = 1e-4f;
    ray.max_distance = INFINITY;
    
    float forwardPDF = 1.0f / (A * pow(dot(ray.direction, camera.forward), 3.0f));
    float3 throughput = float3(1.0f);
    
    return tracePath(ray, throughput, CAMERA_VERTEX, forwardPDF, pixel, uniforms, offset, resources, instances, accelerationStructure, intersectionFunctionTable, areaLights, cameraVertices, MAX_CAMERA_PATH_LENGTH, directLightingContribution, lightTriangles, textureArray, hitLight);
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
                   array<texture2d<float>, MAX_TEXTURES> textureArray,
                   thread float3 &directLightingContribution,
                   thread bool &hitLight
                   )
{
    float4 r = float4(scrambledHalton(offset, 50, uniforms.frameIndex),
                      scrambledHalton(offset, 51, uniforms.frameIndex),
                      scrambledHalton(offset, 52, uniforms.frameIndex),
                      scrambledHalton(offset, 53, uniforms.frameIndex));
    
    lightVertices[0] = sampleAreaLight(areaLights, lightTriangles, uniforms, r);
    float2 randomDirection = float2(scrambledHalton(offset, 40, uniforms.frameIndex), scrambledHalton(offset, 41, uniforms.frameIndex));
    float3 localDirection = sampleCosineWeightedHemisphere(randomDirection);
    float3 worldDirection = normalize(alignHemisphereWithNormal(localDirection, lightVertices[0].normal));
    
    float directionPDF = abs(dot(worldDirection, lightVertices[0].normal)) / M_PI_F;
    
    float epsilon = calculateEpsilon(lightVertices[0].position);
    ray ray;
    ray.origin = lightVertices[0].position + calculateOffset(worldDirection, lightVertices[0].normal, epsilon);
    ray.direction = worldDirection;
    ray.max_distance = INFINITY;
    ray.min_distance = epsilon;

    float forwardPDF = directionPDF; // for L1.forwardPDF
    float3 throughput = lightVertices[0].throughput / forwardPDF;
        
    return tracePath(ray, throughput, LIGHT_VERTEX, forwardPDF, pixel, uniforms, offset, resources, instances, accelerationStructure, intersectionFunctionTable, areaLights, lightVertices, MAX_LIGHT_PATH_LENGTH, directLightingContribution, lightTriangles, textureArray, hitLight);
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

    uint px = min(uint(uv.x * float(uniforms.width)), uniforms.width - 1);
    uint py = min(uint(uv.y * float(uniforms.height)), uniforms.height - 1);
    return uint2(px, py);
}

void splat(texture2d<float, access::read_write> splatTex,
           constant Uniforms& uniforms,
           uint2 pixelCoordinate,
           float3 color,
           device atomic_float* splatBuffer
)
{
    if (pixelCoordinate.x >= uniforms.width || pixelCoordinate.y >= uniforms.height) {
        return;
    }
    
    uint width = uniforms.width;
    uint pixelIndex = (pixelCoordinate.y * width + pixelCoordinate.x) * 3;
        
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 0], color.r, memory_order_relaxed);
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 1], color.g, memory_order_relaxed);
    atomic_fetch_add_explicit(&splatBuffer[pixelIndex + 2], color.b, memory_order_relaxed);
}

float3 connectPaths(thread PathVertex *cameraVertices,
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
                    device atomic_float* splatBuffer,
                    thread bool hitLight,
                    device AreaLight *areaLights,
                    device LightTriangle *lightTriangles
) {
    float3 totalContribution = float3(0.0f);

    for (int c = 1; c <= cameraPathLength; c++) { // c/l = 0 means no vertices there at all
        for (int l = 0; l <= lightPathLength; l++) {
            int depth = c + l - 2;
            
            if ((c == 1 && l == 1) || depth < 0 || depth > MAX_PATH_LENGTH)
                continue;
        
            float3 contribution;;
            
            float4 r4 = float4(scrambledHalton(offset, c * 2 + l + 1, uniforms.frameIndex),
                               scrambledHalton(offset, c * 2 + l + 2, uniforms.frameIndex),
                               scrambledHalton(offset, c * 2 + l + 3, uniforms.frameIndex),
                               scrambledHalton(offset, c * 2 + l + 4, uniforms.frameIndex)
                               );
            
            PathVertex cameraVertex = cameraVertices[c - 1];
            PathVertex lightVertex = l != 1 ? lightVertices[l - 1] : sampleAreaLight(areaLights, lightTriangles, uniforms, r4);
            PathVertex sampled;
                        
            if (l == 0) {
                continue;
                if (c == cameraPathLength and hitLight) {
                    contribution = cameraVertex.throughput * lightVertices[0].material_color; // vertex throughput is up until the point

                    float MISWeight = calculateMISWeight(cameraVertices, lightVertices, c, l, sampled);
//                    DEBUG("c = %d, l = 0 -- MIS: %f", c, MISWeight);
                    totalContribution += contribution * MISWeight;
                }
                continue;
            }
//            continue;
            
            if ((c > 1 and cameraVertex.is_delta) || lightVertex.is_delta)
                continue;
            
            float3 connectionVector = lightVertex.position - cameraVertex.position;
            float connectionDistance = length(connectionVector);
            float3 connectionDirection = connectionVector / connectionDistance;
            
            float cosCamera = dot(connectionDirection, cameraVertex.normal);
            float cosLight = dot(-connectionDirection, lightVertex.normal);
            
            if (cosCamera <= 0.0f || cosLight <= 0.0f)
                continue;
            
            cosCamera = max(1e-6f, cosCamera);
            cosLight = max(1e-6f, cosLight);
            
            if (!isVisible(cameraVertex.position, cameraVertex.normal, connectionDirection, connectionDistance, resources, instances, accelerationStructure, intersectionFunctionTable)) {
                continue;
            }
                                                            
            float d2 = max(connectionDistance * connectionDistance, 0.0f);
            float geometricTerm = (cosCamera * cosLight) / d2;
            
            if (c == 1) {
//                continue;
                float pdf = connectionDistance * connectionDistance / cosCamera;
                float3 cam = 1.0f / (A * pow(cosCamera, 4.0f));
                cam /= pdf;
                float3 brdfLight = evaluateBSDF(lightVertex, cameraVertex);

                contribution = (cam * brdfLight
                                 * lightVertex.throughput * geometricTerm);
            }
            else if (l == 1) {
//                continue;

                sampled = lightVertex;
                
//                debug(A);
                float3 brdfCamera = evaluateBSDF(cameraVertex, lightVertex);
//                debug(sampled.throughput);
                contribution = (cameraVertex.throughput * brdfCamera
                                 * sampled.throughput * geometricTerm);
//                debug(sampled.throughput);
//                debug(contribution);
//
//                DEBUG("cambrdf: %f", brdfCamera.x);

            }
            else {
//                continue;
                sampled = cameraVertex;
                float3 brdfCamera = evaluateBSDF(cameraVertex, lightVertex);
                float3 brdfLight = evaluateBSDF(lightVertex, cameraVertex);
                contribution = (cameraVertex.throughput * brdfCamera
                                 * lightVertex.throughput * brdfLight
                                 * geometricTerm);
//                DEBUG("cambrdf: %f, lightbrdf: %f", brdfCamera.x, brdfLight.x);
            }
            
            float MISWeight = calculateMISWeight(cameraVertices, lightVertices, c, l, sampled);
            contribution *= MISWeight;

//            if (false)
//                DEBUG("c: %d, l: %d, weight: %f, mag: %f, g: %f, cmag: %f, lmag %f",
//                      c, l, MISWeight, length(contribution), geometricTerm, length(cameraVertex.throughput), length(lightVertex.throughput));
//            DEBUG("c: %d, weight: %f", c, MISWeight);
        
//            debug(MISWeight);
            if (c == 1) {
                uint2 pixel = projectToScreen(lightVertex.position, uniforms);
                splat(splatTex, uniforms, pixel, contribution, splatBuffer);
            } else {
                totalContribution += contribution;
            }
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
    cameraRay.min_distance = calculateEpsilon(cameraRay.origin);
    cameraRay.max_distance = INFINITY;
    
    PathVertex cameraVertices[MAX_CAMERA_PATH_LENGTH];
    PathVertex lightVertices[MAX_LIGHT_PATH_LENGTH];
    
    float3 directLightingContribution = float3(0.0f);
    bool hitLight = false;
    
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
                                           textureArray,
                                           hitLight
                                           );
    
#ifndef BDPT
    int lightPathLength = 0;
#else
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
                                         textureArray,
                                         directLightingContribution,
                                         hitLight
                                         );
#endif
        
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
                                         splatBuffer,
                                         hitLight,
                                         areaLights,
                                         lightTriangles
                                         );
    }
    
    float indirectWeight = 1.0f;
    float3 totalLighting = (1.0f * 1.0f * directLightingContribution + 1.0f * 1.0f * indirectLighting * indirectWeight);
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
    finalImage.write(float4(1.0f * totalLighting + 1.0f * 1.0f * totalSplat, 1.0f), tid);
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
