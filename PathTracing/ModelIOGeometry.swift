//
//  ModelIOGeometry.swift
//  PathTracing
//
//  Created on 4/26/25.
//

import Foundation
import MetalKit
import ModelIO

class ModelIOGeometry: Geometry {
    let resolver: MDLAssetResolver?
    
    var lightGeometry: LightGeometry
    
    init(device: MTLDevice, modelURL: URL, textureURL: URL? = transparentURL, defaultColor: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0), inwardsNormals: Bool = false) {
        self.resolver = MDLPathAssetResolver(path: modelURL.deletingLastPathComponent().path)
        lightGeometry = LightGeometry(device: device)
        super.init(device: device)
        
        self.inwardsNormals = inwardsNormals
        
        let bufferAllocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: modelURL,
                             vertexDescriptor: getVertexDescriptor(),
                             bufferAllocator: bufferAllocator
                            )
        
        asset.loadTextures()
        asset.resolver = MDLPathAssetResolver(path: modelURL.deletingLastPathComponent().path)
        print(modelURL.pathComponents.last)
//        print(modelURL.path,asset.childObjects(of: MDLLight.self).count, asset.childObjects(of: MDLAreaLight.self).count, asset.childObjects(of: MDLMesh.self).count)
        for i in 0..<asset.count {
            let object = asset.object(at: i)
            let transform = object.transform?.matrix ?? LinearAlgebra.identity()
            processObject(from: object, device: device, transform: transform, defaultColor: defaultColor)
        }
        
        uploadToBuffers()
    }
    
    private func processObject(from object: MDLObject, device: MTLDevice, transform: matrix_float4x4, defaultColor: SIMD3<Float>) {
        let localTransform = object.transform?.matrix ?? LinearAlgebra.identity()
        let cumulativeTransform = transform * localTransform

        if let mesh = object as? MDLMesh {
            extractMeshData(from: mesh, device: device, transform: cumulativeTransform, defaultColor: defaultColor)
        }

        for child in object.children.objects {
            processObject(from: child, device: device, transform: cumulativeTransform, defaultColor: defaultColor)
        }
    }
    
    private func extractMeshData(from mesh: MDLMesh, device: MTLDevice, transform: matrix_float4x4, defaultColor: SIMD3<Float>) {
        if mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeNormal) == nil {
            mesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.0)
        }
        
        if mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTextureCoordinate) == nil {
            mesh.addUnwrappedTextureCoordinates(forAttributeNamed: MDLVertexAttributeTextureCoordinate)
        }
        
        guard let vertexBuffer = mesh.vertexBuffers.first else { fatalError("Failed getting vertex buffer") }
        let vertexData = vertexBuffer.map().bytes
        
        let vertexStride = (mesh.vertexDescriptor.layouts[0] as! MDLVertexBufferLayout).stride
        let positionAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributePosition)
        let normalAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeNormal)
        let texCoordAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTextureCoordinate)
        let colorAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeColor)
        
        var triangleVertices: [SIMD3<Float>] = []
        var triangleNormals: [SIMD3<Float>] = []
        var triangleTexCoords: [SIMD2<Float>] = []
        var triangleColors: [SIMD3<Float>] = []
        var triangleMaterials: [Material] = []
        
        for submesh in mesh.submeshes! {
            guard let submesh = submesh as? MDLSubmesh else { fatalError("Could not cast to submesh") }
            let indexBuffer = submesh.indexBuffer
            let indexData = indexBuffer.map().bytes
            let indexCount = submesh.indexCount
            let indexType = submesh.indexType

            var submeshMaterial = Material()
            if let mdlMaterial = submesh.material {
                submeshMaterial = extractMaterialData(from: mdlMaterial)
                let baseColor = mdlMaterial.property(with: MDLMaterialSemantic.baseColor)
                let mdlTexture = baseColor?.textureSamplerValue?.texture
                let textureLoader = MTKTextureLoader(device: device)
//                print("color", baseColor?.color, baseColor?.float3Value)
//                let emission = mdlMaterial.property(with: MDLMaterialSemantic.emission)
//                let emissionTexture = emission?.textureSamplerValue?.texture
//                print("emission", emission?.color, emission?.float3Value, emission?.luminance, emission?.textureSamplerValue?.texture?.description, emission?.textureSamplerValue?.texture?.name)
            
                
//                if let emissionProperty = mdlMaterial.property(with: .emission) {
//
//                    if let textureValue = emissionProperty.textureSamplerValue?.texture {
//                        print("texture")
//                    }
//                    if let colorValue = emissionProperty.color {
//                        print("color: \(colorValue)")
//                    }
//                }
                
                
                do {
                    if let cgImage = mdlTexture?.imageFromTexture() {
                        let mtlTexture = try textureLoader.newTexture(cgImage: cgImage.takeRetainedValue())
                        let identifier = "\(submesh.hashValue)+\(mdlTexture.hashValue)"
                        let index = TextureRegistry.shared.addTexture(mtlTexture, identifier: identifier)
                        submeshMaterial.texture_index = UInt32(index)
                    } else {
                        let index = TextureRegistry.shared.getIndex(for: transparentURL.path)
                        submeshMaterial.texture_index = UInt32(index)
                    }
                } catch {
                    print("Failed to convert textures: \(error)")
                }
                
            } else {
                print("Failed to get material")
            }
        
            let emission = submesh.material!.property(with: MDLMaterialSemantic.emission)
            let emissionTexture = emission?.textureSamplerValue?.texture

            let emissionData = emissionTexture?.texelDataWithTopLeftOrigin()
            

            for i in stride(from: 0, to: indexCount, by: 3) {
                triangleMaterials.append(submeshMaterial)
                
                let i0 = getIndexValue(from: indexData, at: i, type: indexType)
                let i1 = getIndexValue(from: indexData, at: i+1, type: indexType)
                let i2 = getIndexValue(from: indexData, at: i+2, type: indexType)
                
                if let positionAttribute = positionAttribute {
                    let offset = positionAttribute.offset
                    
                    let v0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let v1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let v2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    
                    triangleVertices.append(LinearAlgebra.transformPosition(position: v0Ptr.pointee, with: transform))
                    triangleVertices.append(LinearAlgebra.transformPosition(position: v1Ptr.pointee, with: transform))
                    triangleVertices.append(LinearAlgebra.transformPosition(position: v2Ptr.pointee, with: transform))
                }
                
                if let normalAttribute = normalAttribute {
                    let offset = normalAttribute.offset
                    
                    let n0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let n1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let n2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    
                    var n0 = n0Ptr.pointee
                    var n1 = n1Ptr.pointee
                    var n2 = n2Ptr.pointee
                    
                    if inwardsNormals {
                        n0 = -n0
                        n1 = -n1
                        n2 = -n2
                    }
                    
                    triangleNormals.append(LinearAlgebra.transformNormal(normal: n0, with: transform))
                    triangleNormals.append(LinearAlgebra.transformNormal(normal: n1, with: transform))
                    triangleNormals.append(LinearAlgebra.transformNormal(normal: n2, with: transform))
                }
                
                if let texCoordAttribute = texCoordAttribute {
                    let offset = texCoordAttribute.offset
                    
                    let t0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                                         .assumingMemoryBound(to: SIMD2<Float>.self)
                    let t1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                                         .assumingMemoryBound(to: SIMD2<Float>.self)
                    let t2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                                         .assumingMemoryBound(to: SIMD2<Float>.self)

                    triangleTexCoords.append(t0Ptr.pointee)
                    triangleTexCoords.append(t1Ptr.pointee)
                    triangleTexCoords.append(t2Ptr.pointee)
                    if emissionTexture != nil && emissionData != nil {
                        let width = Int(emissionTexture!.dimensions.x)
                        let height = Int(emissionTexture!.dimensions.y)
                        let emissionColor = sampleEmissionTexture(data: emissionData!, w: width, h: height, at: t0Ptr.pointee)
                        if length(emissionColor) > 0.1 {
                            print(emissionColor)
                        }
                    }
                }
                
                if let colorAttribute = colorAttribute {
                    let offset = colorAttribute.offset
                    
                    let c0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let c1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)
                    let c2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                                         .assumingMemoryBound(to: SIMD3<Float>.self)

                    triangleColors.append(c0Ptr.pointee)
                    triangleColors.append(c1Ptr.pointee)
                    triangleColors.append(c2Ptr.pointee)
                } else {
                    var color = defaultColor
                    
                    if let mdlMaterial = submesh.material, let baseColorProperty = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
                        color = baseColorProperty.float3Value

                        if simd_length(color) < 1e-5 {
                            color = SIMD3<Float>(repeating: 1.0)
                        }
                    }
                    
                    triangleColors.append(color)
                    triangleColors.append(color)
                    triangleColors.append(color)
                }
            }
        }
        
        vertices += triangleVertices
        normals += triangleNormals
        texCoords += triangleTexCoords
        colors += triangleColors
        materials += triangleMaterials
    }
    
    private func getIndexValue(from data: UnsafeRawPointer, at offset: Int, type: MDLIndexBitDepth) -> UInt32 {
        switch type {
        case .uInt8:
            return UInt32(data.advanced(by: offset).assumingMemoryBound(to: UInt8.self).pointee)
        case .uInt16:
            return UInt32(data.advanced(by: offset * 2).assumingMemoryBound(to: UInt16.self).pointee)
        case .uInt32:
            return data.advanced(by: offset * 4).assumingMemoryBound(to: UInt32.self).pointee
        default:
            return 0
        }
    }
    
    private func extractMaterialData(from mdlMaterial: MDLMaterial) -> Material {
        var material = Material()
        
        if let opacityProperty = mdlMaterial.property(with: MDLMaterialSemantic.opacity) {
            material.opacity = opacityProperty.floatValue
        } else {
            material.opacity = 1.0
        }
        
        if let refractionProperty = mdlMaterial.property(with: MDLMaterialSemantic.materialIndexOfRefraction) {
            material.refraction = refractionProperty.floatValue
        } else {
            material.refraction = 1.5
        }
        
        if let roughnessProperty = mdlMaterial.property(with: MDLMaterialSemantic.roughness) {
            material.roughness = roughnessProperty.floatValue
        } else {
            material.roughness = 1.0
        }

        if let metallicProperty = mdlMaterial.property(with: MDLMaterialSemantic.metallic) {
            material.metallic = metallicProperty.floatValue
        } else {
            material.metallic = 0.0
        }

        if let emissionProperty = mdlMaterial.property(with: MDLMaterialSemantic.emission) {
            material.emission = emissionProperty.float3Value
        } else {
            material.emission = .zero
        }

        return material
    }
    
    func sampleEmissionTexture(data: Data, w: Int, h: Int, at uv: SIMD2<Float>) -> SIMD3<Float> {
        let u = uv.x.truncatingRemainder(dividingBy: 1.0)
        let v = uv.y.truncatingRemainder(dividingBy: 1.0)

        let px = min(max(Int(u * Float(w)), 0), w - 1)
        let py = min(max(Int((1 - v) * Float(h)), 0), h - 1) // possibly change 1 - v

        let bytesPerPixel = 4
        let rowBytes = bytesPerPixel * w
        let offset = py * rowBytes + px * bytesPerPixel

        let r = Float(data[offset + 0]) / 255
        let g = Float(data[offset + 1]) / 255
        let b = Float(data[offset + 2]) / 255

        return SIMD3<Float>(r, g, b)
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
}

func getVertexDescriptor() -> MDLVertexDescriptor {
    let vertexDescriptor = MTLVertexDescriptor()
    var offset = 0
    
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = offset
    vertexDescriptor.attributes[0].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride
    
    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = offset
    vertexDescriptor.attributes[1].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride
    
    vertexDescriptor.attributes[2].format = .float2
    vertexDescriptor.attributes[2].offset = offset
    vertexDescriptor.attributes[2].bufferIndex = 0
    offset += MemoryLayout<SIMD2<Float>>.stride
    
    vertexDescriptor.layouts[0].stride = offset
    vertexDescriptor.layouts[0].stepFunction = .perVertex
    
    let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(vertexDescriptor)
    (mdlVertexDescriptor.attributes[0] as! MDLVertexAttribute).name = MDLVertexAttributePosition
    (mdlVertexDescriptor.attributes[1] as! MDLVertexAttribute).name = MDLVertexAttributeNormal
    (mdlVertexDescriptor.attributes[2] as! MDLVertexAttribute).name = MDLVertexAttributeTextureCoordinate
    return mdlVertexDescriptor
}
