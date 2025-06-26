//
//  Geometry.swift
//  PathTracing
//
//  Created on 3/21/25.
//

import ModelIO
import MetalKit

class Geometry {
    let device: MTLDevice
    
    var vertexPositionBuffer: MTLBuffer?
    var vertexNormalBuffer: MTLBuffer?
    var vertexColorBuffer: MTLBuffer?
    var textureCoordinatesBuffer: MTLBuffer?
    var materialBuffer: MTLBuffer?
    var vertexTangentBuffer: MTLBuffer?
    var vertexBitangentBuffer: MTLBuffer?

    var vertices: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var colors: [SIMD3<Float>] = []
    var texCoords: [SIMD2<Float>] = []
    var materials: [Material] = []
    var tangents: [SIMD3<Float>] = []
    var bitangents: [SIMD3<Float>] = []
    
    var lightGeometry: LightGeometry?
    var inwardsNormals: Bool = false
        
    init(device: MTLDevice) {
        self.device = device
    }

    func uploadToBuffers() {
        let options = getManagedBufferStorageMode()
        
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
                                              length: tangents.count * MemoryLayout<Material>.stride,
                                              options: options)
            
        }
        
        if !bitangents.isEmpty {
            vertexBitangentBuffer = device.makeBuffer(bytes: bitangents,
                                              length: bitangents.count * MemoryLayout<Material>.stride,
                                              options: options)
            
        }

        #if !os(iOS)
        if let buffer = vertexPositionBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }

        if let buffer = vertexNormalBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }

        if let buffer = vertexColorBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }

        if let buffer = textureCoordinatesBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }

        if let buffer = materialBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }
        
        if let buffer = vertexTangentBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }

        if let buffer = vertexBitangentBuffer {
            buffer.didModifyRange(0..<buffer.length)
        }
        #endif
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
}

class ObjGeometry: Geometry {
    var material: Material?
    var texture: MTLTexture?
    
    init(device: MTLDevice, objURL: URL, textureURL: URL = transparentURL, color: SIMD3<Float> = .one, emissionColor: SIMD3<Float> = .zero, material: Material = PLASTIC, inwardsNormals: Bool = false) {
        super.init(device: device)
        lightGeometry = LightGeometry(device: device)
        
        self.inwardsNormals = inwardsNormals
        self.material = material
        
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
                        // accounting for 1‑based
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
            let index = TextureRegistry.shared.addTexture(texture!, identifier: textureURL.path)
            self.material?.texture_index = Int32(index)
        } catch {
            fatalError("Couldn't load texture: \(error)")
        }
        
        for face in faces {
            for vertex in face {
                let pos = positions[vertex.v]
                let norm = normalsArray[vertex.n]
                let uv = textureCoordinates[vertex.vt]
                
                if let lightGeo = lightGeometry, length(emissionColor) > 0.1 {
                    lightGeo.vertices.append(pos)
                    lightGeo.normals.append(inwardsNormals ? -norm : norm)
                    lightGeo.colors.append(color)
                    lightGeo.texCoords.append(uv)
                    lightGeo.materials.append(self.material!)
                    lightGeo.lightColors.append(emissionColor)
                } else {
                    vertices.append(pos)
                    normals.append(inwardsNormals ? -norm : norm)
                    colors.append(color)
                    texCoords.append(uv)
                    materials.append(self.material!)
                }
            }
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
        if let cb = vertexColorBuffer { resourceArray.append(cb) }
        if let mb = materialBuffer { resourceArray.append(mb) }
        if let tx = textureCoordinatesBuffer { resourceArray.append(tx) }

        return resourceArray
    }
        
    override func getLightGeometry() -> LightGeometry? {
        return lightGeometry
    }
}

class GeometryInstance {
    let geometry: Geometry
    
    var translation: SIMD3<Float>
    var rotation: SIMD3<Float>
    var scale: SIMD3<Float>
    
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
        return MTLPackedFloat4x3(columns: (
            MTLPackedFloat3Make(transform[0][0], transform[0][1], transform[0][2]),
            MTLPackedFloat3Make(transform[1][0], transform[1][1], transform[1][2]),
            MTLPackedFloat3Make(transform[2][0], transform[2][1], transform[2][2]),
            MTLPackedFloat3Make(transform[3][0], transform[3][1], transform[3][2])
        ))
    }
    
    func getLightTriangles() -> ([LightTriangle], Float, SIMD3<Float>) {
        var lightTriangles: [LightTriangle] = []
        var totalArea: Float = 0.0
        var averageEmission = SIMD3<Float>(0.0, 0.0, 0.0)
        
        let vertices = geometry.vertices
        guard let lightColors = (geometry as? LightGeometry)?.lightColors,
              let lightAmplifier = (geometry as? LightGeometry)?.lightAmplifier else { fatalError("Could not get light geometry") }
        
        for i in stride(from: 0, to: vertices.count, by: 3) {
            if i + 2 < vertices.count {
                let worldV0 = transform * SIMD4<Float>(vertices[i], 1.0)
                let worldV1 = transform * SIMD4<Float>(vertices[i + 1], 1.0)
                let worldV2 = transform * SIMD4<Float>(vertices[i + 2], 1.0)
                
                let v0 = SIMD3<Float>(worldV0.x, worldV0.y, worldV0.z)
                let v1 = SIMD3<Float>(worldV1.x, worldV1.y, worldV1.z)
                let v2 = SIMD3<Float>(worldV2.x, worldV2.y, worldV2.z)
                
                let edge1 = v1 - v0
                let edge2 = v2 - v0
                
                let area = 0.5 * simd_length(cross(edge1, edge2))
                totalArea += area
                
                let emission0 = lightColors[i] * lightAmplifier
                let emission1 = lightColors[i + 1] * lightAmplifier
                let emission2 = lightColors[i + 2] * lightAmplifier
                
                averageEmission += emission0 + emission1 + emission2
                
                let triangle = LightTriangle(v0: v0, v1: v1, v2: v2,
                                             emission0: emission0, emission1: emission1, emission2: emission2,
                                             area: area, cdf: 0.0)
                
                lightTriangles.append(triangle)
            }
        }
        
        var cumulativeArea: Float = 0.0
        for i in 0..<lightTriangles.count {
            cumulativeArea += lightTriangles[i].area
            lightTriangles[i].cdf = cumulativeArea / totalArea
        }
        print(totalArea)
        return (lightTriangles, totalArea, averageEmission / Float(lightColors.count))
    }
}
