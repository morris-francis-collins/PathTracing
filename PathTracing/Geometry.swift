//
//  Geometry.swift
//  PathTracing
//
//  Created on 3/21/25.
//

import ModelIO
import MetalKit
import Foundation

struct AreaLight {
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
    
    var areaLights: [AreaLight] = []
    var lightGeometry: LightGeometry?
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
 
//        #if !os(iOS)
//        if let buffer = vertexPositionBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = vertexNormalBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = vertexColorBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = textureCoordinatesBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = materialBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = vertexTangentBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = vertexBitangentBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        if let buffer = primitiveLightIndicesBuffer {
//            buffer.didModifyRange(0..<buffer.length)
//        }
//
//        #endif
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
    
    func getLightGeometry() -> LightGeometry? {
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

class ObjGeometry: Geometry {
    var material: Material!
    var texture: MTLTexture?
    
    init(device: MTLDevice, objURL: URL, textureURL: URL = transparentURL, color: SIMD3<Float> = .one, emissionColor: SIMD3<Float> = .zero, material: Material = PLASTIC, inwardsNormals: Bool = false) {
        super.init(device: device)
        lightGeometry = LightGeometry(device: device)
        
        self.inwardsNormals = inwardsNormals
        self.material = material
        self.material.emissionValue = emissionColor
        self.material.emissionTextureIndex = -1
        
        guard let fileContent = try? String(contentsOf: objURL, encoding: .utf8) else {
            print("Failed to read OBJ file from \(objURL)")
            return
        }
        
        var positions: [SIMD3<Float>] = []
        var normalsArray: [SIMD3<Float>] = []
        var textureCoordinates: [SIMD2<Float>] = []
        var faces: [[(v: Int, vt: Int, n: Int)]] = []
        
        let lines = fileContent.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("v ") {
                // vertex positions
                let parts = trimmed.split(separator: " ")
                if parts.count >= 4, let x = Float(parts[1]), let y = Float(parts[2]), let z = Float(parts[3]) {
                    positions.append(SIMD3<Float>(x, y, z))
                }
            } else if trimmed.hasPrefix("vt ") {
                // texture coordinates
                let parts = trimmed.split(separator: " ")
                if parts.count >= 3,
                   let u = Float(parts[1]),
                   let v = Float(parts[2]) {
                    textureCoordinates.append(SIMD2<Float>(u * 5, v * 5)) // hardcoded for checkerboard
                }
            } else if trimmed.hasPrefix("vn ") {
                // vertex normals
                let parts = trimmed.split(separator: " ")
                if parts.count >= 4,
                   let x = Float(parts[1]),
                   let y = Float(parts[2]),
                   let z = Float(parts[3]) {
                    normalsArray.append(SIMD3<Float>(x, y, z))
                }
            } else if trimmed.hasPrefix("f ") {
                // face definitions
                let parts = trimmed.split(separator: " ")
                var faceVertices: [(v: Int, vt: Int, n: Int)] = []

                for token in parts.dropFirst() {
                    let subTokens = token.split(separator: "/")
                    if subTokens.count >= 3,
                       let vIndex = Int(subTokens[0]),
                       let vtIndex = Int(subTokens[1]),
                       let nIndex = Int(subTokens[2]) {
                        // accounting for 1â€‘based
                        faceVertices.append((v: vIndex - 1, vt: vtIndex - 1, n: nIndex - 1))
                    }
                }

                if faceVertices.count >= 3 {
                    // triangulation: (v0, v1, v2), (v0, v2, v3), etc.
                    for i in 1..<(faceVertices.count - 1) {
                        faces.append([faceVertices[0], faceVertices[i], faceVertices[i + 1]])
                    }
                }
            }
        }
        
                
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false
        ]
        do {
            texture = try textureLoader.newTexture(URL: textureURL, options: options)
            let index = TextureRegistry.shared.addTexture(texture!, identifier: String(TextureRegistry.shared.getTextures().count))
            self.material.colorTextureIndex = Int32(index)
            print("TEXINDEX", index, textureURL.path())
        } catch {
            fatalError("Couldn't load texture: \(error)")
        }
        
        let isEmissive = simd_length_squared(emissionColor) > 1e-4
        let primitiveLightIndex = Int32(isEmissive ? 0 : -1)
        var totalArea: Float = 0.0
        var lightTriangles: [LightTriangle] = []
        
        for face in faces {
            for vertex in face {
                let pos = positions[vertex.v]
                let norm = normalsArray[vertex.n]
                let uv = textureCoordinates[vertex.vt]
                
                vertices.append(pos)
                normals.append(inwardsNormals ? -norm : norm)
                texCoords.append(uv)
                materials.append(self.material)
                primitiveLightIndices.append(primitiveLightIndex)
            }
            
            if isEmissive {
                let i0 = vertices.count - 1
                let i1 = vertices.count - 2
                let i2 = vertices.count - 3

                let v0 = vertices[i0]
                let v1 = vertices[i1]
                let v2 = vertices[i2]
                let e0 = v1 - v0
                let e1 = v2 - v0

                let area = 0.5 * length(simd_cross(e0, e1))
                totalArea += area
                lightTriangles.append(LightTriangle(v0: v0, v1: v1, v2: v2,
                                                    uv0: texCoords[i0], uv1: texCoords[i1], uv2: texCoords[i2],
                                                    emission: self.material.emissionValue,
                                                    emissionTextureIndex: self.material.emissionTextureIndex,
                                                    CDF: totalArea)
                )
            }
        }

        if isEmissive {
            for i in 0..<lightTriangles.count {
                lightTriangles[i].CDF /= totalArea
            }
            areaLights.append(AreaLight(emission: self.material.emissionValue,
                                        emissionTextureIndex: self.material.emissionTextureIndex,
                                        averageEmission: emissionColor,
                                        vertices: vertices,
                                        UVs: texCoords)
            )
        }
        
        uploadToBuffers()
    }
        
    override func geometryDescriptor() -> MTLAccelerationStructureGeometryDescriptor? {
        let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        descriptor.vertexBuffer = vertexPositionBuffer
        descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        descriptor.triangleCount = vertices.count / 3
        return descriptor
    }
        
    override func resources() -> [MTLResource] {
        var resourceArray: [MTLResource] = []
        
        if let nb = vertexNormalBuffer { resourceArray.append(nb) }
        if let mb = materialBuffer { resourceArray.append(mb) }
        if let tx = textureCoordinatesBuffer { resourceArray.append(tx) }
        if let pi = primitiveLightIndicesBuffer { resourceArray.append(pi) }

        return resourceArray
    }
        
    override func getLightGeometry() -> LightGeometry? {
        return lightGeometry
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
    
    let mask: UInt32

    init(geometry: Geometry, translation: SIMD3<Float> = .zero, rotation: SIMD3<Float> = .zero, scale: SIMD3<Float> = .one, mask: UInt32) {
        self.geometry = geometry
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.mask = mask
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
