//
//  PathTracing.h
//  PathTracing
//
//  Created on 4/11/26.
//

#include "Utility.h"
#include "Lights.h"
#include "Interactions.h"
#include "Materials.h"
#include "Samplers.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

float3 sampleNEE(float3 throughput,
                 ray ray,
                        
                 constant Light* lights,
                 constant LightTriangle* lightTriangles,
                 thread Sampler& sampler,
                 constant Textures* textures,
                 constant AliasEntry* lightAliasEntries,
                 constant AliasEntry* lightTriangleAliasEntries,
                 texture2d<float> environmentMapTexture,
                 constant AliasEntry* environmentMapAliasEntries,
                        
                 thread SurfaceInteraction& si,

                 constant Uniforms& uniforms,
                 device MTLAccelerationStructureInstanceDescriptor* instances,
                 instance_acceleration_structure accelerationStructure);

#endif
