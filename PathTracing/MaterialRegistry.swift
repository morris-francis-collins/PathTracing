//
//  MaterialRegistry.swift
//  PathTracing
//
//

import MetalKit

extension VectorParameter: Equatable {
    public static func ==(lhs: VectorParameter, rhs: VectorParameter) -> Bool {
        return lhs.value.x == rhs.value.x &&
               lhs.value.y == rhs.value.y &&
               lhs.value.z == rhs.value.z &&
               lhs.textureIndex == rhs.textureIndex
    }
}

extension ScalarParameter: Equatable {
    public static func ==(lhs: ScalarParameter, rhs: ScalarParameter) -> Bool {
        return lhs.value == rhs.value && lhs.textureIndex == rhs.textureIndex
    }
}

extension Material: Equatable {
    public static func ==(lhs: Material, rhs: Material) -> Bool {
        return lhs.color == rhs.color &&
               lhs.refraction == rhs.refraction &&
               lhs.roughness == rhs.roughness &&
               lhs.metallic == rhs.metallic &&
               lhs.emission == rhs.emission &&
               lhs.BXDFs == rhs.BXDFs
    }
}

class MaterialRegistry {
    static let shared = MaterialRegistry()
    
    private var materials: [Material] = []
    private var materialBuffer: MTLBuffer?
        
    func addMaterial(_ material: Material) -> Int {
        let index = getIndex(for: material)
        
        if (index != -1) {
            return index;
        }
        
        materials.append(material)
        return materials.count - 1
    }
    
    func getIndex(for material: Material) -> Int {
        for i in 0..<materials.count {
            if material == materials[i] {
                return i
            }
        }
        return -1;
    }
    
    func getMaterials() -> [Material] {
        return materials
    }
    
    func uploadToBuffers(device: MTLDevice) {
        if !materials.isEmpty {
            materialBuffer = device.makeBuffer(bytes: materials,
                                               length: materials.count * MemoryLayout<Material>.stride,
                                               options: getManagedBufferStorageMode())
        }
    }
    
    func getBuffer() -> MTLBuffer? {
        return materialBuffer
    }
}
