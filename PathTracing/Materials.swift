//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

import simd

func createStaticMaterial(color: SIMD3<Float>, refraction: Float, roughness: Float, metallic: Float, BXDFs: Int32) -> Material {
    return Material(color: VectorParameter(value: color, textureIndex: -1),
                    refraction: ScalarParameter(value: refraction, textureIndex: -1),
                    roughness: ScalarParameter(value: roughness, textureIndex: -1),
                    metallic: ScalarParameter(value: metallic, textureIndex: -1),
                    BXDFs: BXDFs
                    )
}

var GLASS = createStaticMaterial(color: .one, refraction: 1.5, roughness: 0.0, metallic: 0.0, BXDFs: SPECULAR_TRANSMISSION)
var PLASTIC = createStaticMaterial(color: 0.7 * .one, refraction: 1, roughness: 0.9, metallic: 0.0, BXDFs: DIFFUSE)
var MIRROR = createStaticMaterial(color: .one, refraction: 1.5, roughness: 0.0, metallic: 1.0, BXDFs: CONDUCTOR)
var WATER = createStaticMaterial(color: .one, refraction: 1.3, roughness: 0.0, metallic: 0.0, BXDFs: SPECULAR_TRANSMISSION)

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
