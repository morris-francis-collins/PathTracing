//
//  MaterialRegistry.swift
//  PathTracing
//
//

import MetalKit

extension Material: Equatable {
    public static func ==(lhs: Material, rhs: Material) -> Bool {
        return lhs.colorValue == rhs.colorValue &&
               lhs.colorTextureIndex == rhs.colorTextureIndex &&
               lhs.roughnessValue == rhs.roughnessValue &&
               lhs.roughnessTextureIndex == rhs.roughnessTextureIndex &&
               lhs.metallicValue == rhs.metallicValue &&
               lhs.metallicTextureIndex == rhs.metallicTextureIndex &&
               lhs.emissionValue == rhs.emissionValue &&
               lhs.emissionTextureIndex == rhs.emissionTextureIndex &&
               lhs.emissiveStrength == rhs.emissiveStrength &&
               lhs.ior == rhs.ior &&
               lhs.alphaMode == rhs.alphaMode &&
               lhs.alphaCutoff == rhs.alphaCutoff &&
               lhs.transmissionValue == rhs.transmissionValue &&
               lhs.transmissionTextureIndex == rhs.transmissionTextureIndex &&
               lhs.thicknessFactor == rhs.thicknessFactor &&
               lhs.normalTextureIndex == rhs.normalTextureIndex &&
               lhs.clearcoatValue == rhs.clearcoatValue &&
               lhs.clearcoatTextureIndex == rhs.clearcoatTextureIndex &&
               lhs.clearcoatRoughnessValue == rhs.clearcoatRoughnessValue &&
               lhs.clearcoatRoughnessTextureIndex == rhs.clearcoatRoughnessTextureIndex &&
               lhs.clearcoatNormalTextureIndex == rhs.clearcoatNormalTextureIndex &&
               lhs.doubleSided == rhs.doubleSided
    }
}

class MaterialRegistry {
    static let shared = MaterialRegistry()
    
    private var materials: [Material] = []
    private var materialBuffer: MTLBuffer?
    
    func addMaterial(_ material: Material) -> Int {
        if let index = materials.firstIndex(of: material) {
            return index
        }
        
        materials.append(material)
        return materials.count - 1
    }
    
    func getMaterials() -> [Material] {
        return materials
    }
    
    func uploadToBuffers(device: MTLDevice) {
        if !materials.isEmpty {
            materialBuffer = device.makeBuffer(
                bytes: materials,
                length: materials.count * MemoryLayout<Material>.stride,
                options: getManagedBufferStorageMode()
            )
        }
    }

    func getBuffer() -> MTLBuffer? {
        return materialBuffer
    }
}
