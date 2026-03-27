//
//  Materials.swift
//  PathTracing
//
//  Created on 4/11/25.
//

import simd

func createStaticMaterial(color: SIMD3<Float> = SIMD3<Float>(0.8, 0.8, 0.8),
                          ior: Float = 1.5,
                          roughness: Float = 0.5,
                          metallic: Float = 0.0,
                          emission: SIMD3<Float> = .zero,
                          transmission: Float = 0.0) -> Material
{
    var material = Material()
    material.colorValue = color
    material.colorTextureIndex = -1
    material.roughnessValue = roughness
    material.roughnessTextureIndex = -1
    material.metallicValue = metallic
    material.metallicTextureIndex = -1
    material.emissionValue = emission
    material.emissionTextureIndex = -1
    material.emissiveStrength = 1.0
    material.ior = ior
    material.alphaMode = 0
    material.alphaCutoff = 0.5
    material.transmissionValue = transmission
    material.transmissionTextureIndex = -1
    material.thicknessFactor = transmission > 0.01 ? 1.0 : 0.0
    material.attenuationColor = SIMD3<Float>(1, 1, 1)
    material.attenuationDistance = .infinity
    material.normalTextureIndex = -1
    material.normalScale = 1.0
    material.clearcoatValue = 0
    material.clearcoatTextureIndex = -1
    material.clearcoatRoughnessValue = 0
    material.clearcoatRoughnessTextureIndex = -1
    material.clearcoatNormalTextureIndex = -1
    material.anisotropyStrength = 0
    material.anisotropyRotation = 0
    material.anisotropyTextureIndex = -1
    material.sheenColor = .zero
    material.sheenRoughness = 0
    material.doubleSided = 0
    return material
}

func createEmissiveMaterial(color: SIMD3<Float>) -> Material {
    return createStaticMaterial(color: .one, emission: color)
}

func colorMaterial(material: Material, color: SIMD3<Float>) -> Material {
    var m = material
    m.colorValue = color
    return m
}

let GLASS = createStaticMaterial(color: .one, ior: 1.5, roughness: 0.0, metallic: 0.0, transmission: 1.0)
let PLASTIC = createStaticMaterial(color: 0.7 * .one, ior: 1.0, roughness: 0.9, metallic: 0.0)
let SMOOTH_OPAQUE_DIELECTRIC = createStaticMaterial(color: 0.7 * .one, ior: 1.5, roughness: 0.0, metallic: 0.0)
let MIRROR = createStaticMaterial(color: .one, ior: 1.5, roughness: 0.0, metallic: 1.0)
let WATER = createStaticMaterial(color: .one, ior: 1.33, roughness: 0.0, metallic: 0.0, transmission: 1.0)
let EMISSIVE = createStaticMaterial(color: .one, emission: .one)

let RED = SIMD3<Float>(1.0, 0.0, 0.0)
let ORANGE = SIMD3<Float>(1.0, 0.647, 0.0)
let YELLOW = SIMD3<Float>(1.0, 1.0, 0.0)
let GREEN = SIMD3<Float>(0.0, 1.0, 0.0)
let BLUE = SIMD3<Float>(0.0, 0.0, 1.0)
let INDIGO = SIMD3<Float>(0.294, 0.0, 0.510)
let VIOLET = SIMD3<Float>(0.933, 0.510, 0.933)

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
