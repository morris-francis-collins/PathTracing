//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

var GLASS = Material(opacity: 0.0, refraction: 1.5, roughness_x: 0.0, roughness_y: 0.0, metallic: 0.0, absorption: 0.4 * SIMD3<Float>(1, 0.361, 0.576), texture_index: -1)
var PLASTIC = Material(opacity: 1.0, refraction: 1.45, roughness_x: 0.9, roughness_y: 0.9, metallic: 0.0, absorption: .zero, texture_index: -1)
var MIRROR = Material(opacity: 1.0, refraction: 1.5, roughness_x: 0, roughness_y: 0, metallic: 1.0, absorption: .zero, texture_index: -1)
var GLOSSY_PLASTIC = Material(opacity: 1.0, refraction: 1.45, roughness_x: 0.10, roughness_y: 0.10, metallic: 0.0, absorption: .zero, texture_index: -1)
var GLOSSY_METAL = Material(opacity: 1.0, refraction: 2.5, roughness_x: 0.10, roughness_y: 0.10, metallic: 1.0, absorption: .zero, texture_index: -1)
var WATER = Material(opacity: 0.0, refraction: 2.0, roughness_x: 0.0, roughness_y: 0.0, metallic: 0.0, absorption: SIMD3<Float>(0.1, 0.1, 0), texture_index: -1)

let colors: [SIMD3<Float>] = [
    SIMD3(1.000, 0.000, 0.000),
    SIMD3(1.000, 0.458, 0.000),
    SIMD3(1.000, 0.917, 0.000),
    SIMD3(0.625, 1.000, 0.000),
    SIMD3(0.167, 1.000, 0.000),
    SIMD3(0.000, 1.000, 0.292),
    SIMD3(0.000, 1.000, 0.750),
    SIMD3(0.000, 0.792, 1.000),
    SIMD3(0.000, 0.333, 1.000),
    SIMD3(0.125, 0.000, 1.000),
    SIMD3(0.583, 0.000, 1.000)
]
