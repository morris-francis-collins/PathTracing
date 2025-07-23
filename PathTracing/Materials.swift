//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

var GLASS = Material(color: .one, refraction: 1.5, roughness: 0.0, metallic: 0.0, BXDFs: SPECULAR_TRANSMISSION, textureIndex: -1)
var PLASTIC = Material(color: 0.7 * .one, refraction: 1.5, roughness: 0.9, metallic: 0.0, BXDFs: DIFFUSE, textureIndex: -1)
var MIRROR = Material(color: .one, refraction: 1.5, roughness: 0.0, metallic: 1.0, BXDFs: CONDUCTOR, textureIndex: -1)
var WATER = Material(color: .one, refraction: 1.3, roughness: 0.0, metallic: 0.0, BXDFs: SPECULAR_TRANSMISSION, textureIndex: -1)

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
