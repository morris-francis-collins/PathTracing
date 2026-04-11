//
//  Geometry.swift
//  PathTracing
//
//  Created on 3/21/25.
//

import ModelIO
import MetalKit
import Foundation

struct AreaLightData {
    let emission: SIMD3<Float>
    let emissionTextureIndex: Int32
    let averageEmission: SIMD3<Float>
    let vertices: [SIMD3<Float>]
    let UVs: [SIMD2<Float>]
}

struct TextureInfo {
    let textureURL: URL?
    let uvMultiplier: Float
    
    init(textureURL: URL? = nil, uvMultiplier: Float = 1.0) {
        self.textureURL = textureURL
        self.uvMultiplier = uvMultiplier
    }
}

class Geometry {
    let device: MTLDevice
    
    var vertexPositionBuffer: MTLBuffer?
    var vertexNormalBuffer: MTLBuffer?
    var vertexColorBuffer: MTLBuffer?
    var textureCoordinatesBuffer: MTLBuffer?
    var materialBuffer: MTLBuffer?
    var vertexTangentBuffer: MTLBuffer?
    var vertexBitangentBuffer: MTLBuffer?
    var primitiveLightIndicesBuffer: MTLBuffer?
    var primitiveTriangleDataBuffer: MTLBuffer?

    var vertices: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var colors: [SIMD3<Float>] = []
    var texCoords: [SIMD2<Float>] = []
    var materials: [Material] = []
    var tangents: [SIMD3<Float>] = []
    var bitangents: [SIMD3<Float>] = []
    var primitiveLightIndices: [Int32] = []
    var primitiveTriangleData: [PrimitiveData] = []
    
    var areaLights: [AreaLightData] = []
    var inwardsNormals: Bool = false
        
    init(device: MTLDevice) {
        self.device = device
    }

    func uploadToBuffers() {
        let options: MTLResourceOptions = getManagedBufferStorageMode()
        
        if !vertices.isEmpty {
            vertexPositionBuffer = device.makeBuffer(bytes: vertices,
                                                   length: vertices.count * MemoryLayout<SIMD3<Float>>.stride,
                                                   options: options)
        }
        
        if !normals.isEmpty {
            vertexNormalBuffer = device.makeBuffer(bytes: normals,
                                                 length: normals.count * MemoryLayout<SIMD3<Float>>.stride,
                                                 options: options)
        }
        
        if !texCoords.isEmpty {
            textureCoordinatesBuffer = device.makeBuffer(bytes: texCoords,
                                                       length: texCoords.count * MemoryLayout<SIMD2<Float>>.stride,
                                                       options: options)
        }
        
        if !colors.isEmpty {
            vertexColorBuffer = device.makeBuffer(bytes: colors,
                                                length: colors.count * MemoryLayout<SIMD3<Float>>.stride,
                                                options: options)
        }
        
        if !materials.isEmpty {
            materialBuffer = device.makeBuffer(bytes: materials,
                                              length: materials.count * MemoryLayout<Material>.stride,
                                              options: options)
            
        }
        
        if !tangents.isEmpty {
            vertexTangentBuffer = device.makeBuffer(bytes: tangents,
                                              length: tangents.count * MemoryLayout<SIMD3<Float>>.stride,
                                              options: options)
            
        }
        
        if !bitangents.isEmpty {
            vertexBitangentBuffer = device.makeBuffer(bytes: bitangents,
                                              length: bitangents.count * MemoryLayout<SIMD3<Float>>.stride,
                                              options: options)
            
        }
        
        if !primitiveLightIndices.isEmpty {
            primitiveLightIndicesBuffer = device.makeBuffer(bytes: primitiveLightIndices,
                                              length: primitiveLightIndices.count * MemoryLayout<Int32>.stride,
                                              options: options)
    
        }
        
        if !primitiveTriangleData.isEmpty {
            primitiveTriangleDataBuffer = device.makeBuffer(bytes: primitiveTriangleData,
                                              length: primitiveTriangleData.count * MemoryLayout<PrimitiveData>.stride,
                                              options: options)
        }
     }

    func geometryDescriptor() -> MTLAccelerationStructureGeometryDescriptor? {
        return nil
    }

    func resources() -> [MTLResource] {
        return []
    }
    
    func intersectionFunctionName() -> String? {
        return nil
    }
    
    func encodeResources(to encoder: MTLArgumentEncoder) {
        if let nb = vertexNormalBuffer {
            encoder.setBuffer(nb, offset: 0, index: 0)
        }
        if let mb = materialBuffer {
            encoder.setBuffer(mb, offset: 0, index: 1)
        }
        if let tx = textureCoordinatesBuffer {
            encoder.setBuffer(tx, offset: 0, index: 2)
        }
        if let pi = primitiveLightIndicesBuffer {
            encoder.setBuffer(pi, offset: 0, index: 3)
        }
    }
}

class GeometryInstance {
    let geometry: Geometry
    
    let translation: SIMD3<Float>
    let rotation: SIMD3<Float>
    let scale: SIMD3<Float>
    
    var transform: simd_float4x4 {
        return LinearAlgebra.translate(translation: translation) *
               LinearAlgebra.rotate(eulers: rotation) *
               LinearAlgebra.scale(scale: scale)
    }
    
    init(geometry: Geometry, translation: SIMD3<Float> = .zero, rotation: SIMD3<Float> = .zero, scale: SIMD3<Float> = .one) {
        self.geometry = geometry
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
    }
    
    func getPackedTransform() -> MTLPackedFloat4x3 {
        return MTLPackedFloat4x3(
            MTLPackedFloat3Make(transform[0][0], transform[0][1], transform[0][2]),
            MTLPackedFloat3Make(transform[1][0], transform[1][1], transform[1][2]),
            MTLPackedFloat3Make(transform[2][0], transform[2][1], transform[2][2]),
            MTLPackedFloat3Make(transform[3][0], transform[3][1], transform[3][2])
        )
    }
}
