//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

var GLASS = Material(opacity: 0.05, refraction: 1.6, roughness: 0.05, metallic: 0.0, emission: SIMD3<Float>(repeating: 0.0), texture_index: 0)
var PLASTIC = Material(opacity: 1.0, refraction: 1.45, roughness: 0.95, metallic: 0.0, emission: SIMD3<Float>(repeating: 0.0), texture_index: 0)
var MIRROR = Material(opacity: 1.0, refraction: 2.5, roughness: 0.0, metallic: 1.0, emission: SIMD3<Float>(repeating: 0.0), texture_index: 0)
var GLOSSY_PLASTIC = Material(opacity: 1.0, refraction: 1.45, roughness: 0.10, metallic: 0.0, emission: SIMD3<Float>(repeating: 0.0), texture_index: 0)
var GLOSSY_METAL = Material(opacity: 1.0, refraction: 2.5, roughness: 0.10, metallic: 1.0, emission: SIMD3<Float>(repeating: 0.0), texture_index: 0)
