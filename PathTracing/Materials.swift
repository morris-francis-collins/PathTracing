//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

import simd

func createStaticMaterial(color: SIMD3<Float>, refraction: Float, roughness: Float, metallic: Float, emission: SIMD3<Float>, BXDFs: Int32) -> Material {
    return Material(color: VectorParameter(value: color, textureIndex: -1),
                    refraction: ScalarParameter(value: refraction, textureIndex: -1),
                    roughness: ScalarParameter(value: roughness, textureIndex: -1),
                    metallic: ScalarParameter(value: metallic, textureIndex: -1),
                    emission: VectorParameter(value: emission, textureIndex: -1),
                    BXDFs: BXDFs
                    )
}

func createEmissiveMaterial(color: SIMD3<Float>) -> Material {
    return createStaticMaterial(color: .one, refraction: 1.0, roughness: 0.0, metallic: 0.0, emission: color, BXDFs: DIFFUSE)
}

func colorMaterial(material: Material, color: SIMD3<Float>) -> Material {
    var newMaterial = material
    newMaterial.color.value = color
    return newMaterial
}

let GLASS = createStaticMaterial(color: .one, refraction: 1.5, roughness: 0.0, metallic: 0.0, emission: .zero, BXDFs: SPECULAR_TRANSMISSION)
let PLASTIC = createStaticMaterial(color: 0.7 * .one, refraction: 1, roughness: 0.9, metallic: 0.0, emission: .zero, BXDFs: DIFFUSE)
let MIRROR = createStaticMaterial(color: .one, refraction: 1.5, roughness: 0.0, metallic: 1.0, emission: .zero, BXDFs: CONDUCTOR)
let WATER = createStaticMaterial(color: .one, refraction: 1.3, roughness: 0.0, metallic: 0.0, emission: .zero, BXDFs: SPECULAR_TRANSMISSION)
let EMISSIVE = createStaticMaterial(color: .one, refraction: 1, roughness: 0.0, metallic: 0.0, emission: .one, BXDFs: SPECULAR_TRANSMISSION)

let RED = SIMD3<Float>(1.0, 0.0, 0.0)
let ORANGE = SIMD3<Float>(1.0, 0.647, 0.0)
let YELLOW = SIMD3<Float>(1.0, 1.0, 0.0)
let GREEN = SIMD3<Float>(0.0, 1.0, 0.0)
let BLUE = SIMD3<Float>(0.0, 0.0, 1.0)
let INDIGO = SIMD3<Float>(0.294, 0.0, 0.510)
let BIOLET = SIMD3<Float>(0.933, 0.510, 0.933)

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
