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
    var resolver: MDLAssetResolver?
    let defaultMaterial: Material?
    let textureURL: URL?
    let emissionColor: SIMD3<Float>
    
    init(device: MTLDevice, modelURL: URL, textureURL: URL? = nil, defaultColor: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0), defaultMaterial: Material? = nil, emissionColor: SIMD3<Float> = .zero, inwardsNormals: Bool = false) {
        self.defaultMaterial = defaultMaterial
        self.textureURL = textureURL
        self.emissionColor = emissionColor
        
        super.init(device: device)
        
        lightGeometry = LightGeometry(device: device)

        self.inwardsNormals = inwardsNormals
        
        let bufferAllocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: modelURL,
                             vertexDescriptor: getVertexDescriptor(),
                             bufferAllocator: bufferAllocator
                            )
        
        asset.loadTextures()
        resolver = asset.resolver!

//        print(modelURL.pathComponents.last)
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
        
        mesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate, normalAttributeNamed: MDLVertexAttributeNormal, tangentAttributeNamed: MDLVertexAttributeTangent)

        guard let vertexBuffer = mesh.vertexBuffers.first else { fatalError("Failed getting vertex buffer") }
        let vertexData = vertexBuffer.map().bytes
        
        let vertexStride = (mesh.vertexDescriptor.layouts[0] as! MDLVertexBufferLayout).stride
        let positionAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributePosition)
        let normalAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeNormal)
        let texCoordAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTextureCoordinate)
        let colorAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeColor)
        let tangentAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeTangent)
        let bitangentAttribute = mesh.vertexDescriptor.attributeNamed(MDLVertexAttributeBitangent)

        
        var triangleVertices: [SIMD3<Float>] = []
        var triangleNormals: [SIMD3<Float>] = []
        var triangleTexCoords: [SIMD2<Float>] = []
        var triangleColors: [SIMD3<Float>] = []
        var triangleMaterials: [Material] = []
        var triangleTangents: [SIMD3<Float>] = []
        var triangleBitangents: [SIMD3<Float>] = []
                
        for submesh in mesh.submeshes! {
            guard let submesh = submesh as? MDLSubmesh else { fatalError("Could not cast to submesh") }
            let indexBuffer = submesh.indexBuffer
            let indexData = indexBuffer.map().bytes
            let indexCount = submesh.indexCount
            let indexType = submesh.indexType
            
            var submeshMaterial = Material()
            submeshMaterial.texture_index = -1
            var currentColor = defaultColor
        
            if let mdlMaterial = submesh.material {
                mdlMaterial.loadTextures(using: resolver!)
                
                if let baseColorProperty = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
                    if baseColorProperty.type == .float3 {
                        currentColor = baseColorProperty.float3Value
                    }
                }

                if let opacityProperty = mdlMaterial.property(with: MDLMaterialSemantic.opacity) {
                    if opacityProperty.type == .float {
                        submeshMaterial.opacity = opacityProperty.floatValue
                    }
                } else {
                    submeshMaterial.opacity = 1.0
                }

                
                if let refractionProperty = mdlMaterial.property(with: MDLMaterialSemantic.materialIndexOfRefraction) {
                    if refractionProperty.type == .float {
                        submeshMaterial.refraction = refractionProperty.floatValue
                    }
                } else {
                    submeshMaterial.refraction = 1.5
                }

                if let roughnessProperty = mdlMaterial.property(with: MDLMaterialSemantic.roughness) {
                    if roughnessProperty.type == .float {
                        submeshMaterial.roughness_x = roughnessProperty.floatValue
                        submeshMaterial.roughness_y = roughnessProperty.floatValue
                    }
                } else {
                    submeshMaterial.roughness_x = 0.0
                    submeshMaterial.roughness_y = 0.0
                }
                
                if let metallicProperty = mdlMaterial.property(with: MDLMaterialSemantic.metallic) {
                    if metallicProperty.type == .float {
                        submeshMaterial.metallic = metallicProperty.floatValue
                    }
                } else {
                    submeshMaterial.metallic = 0.0
                }
                
                if let defaultMaterial = self.defaultMaterial {
                    submeshMaterial = defaultMaterial
                }
                
                let baseColor = mdlMaterial.property(with: MDLMaterialSemantic.baseColor)
                let mdlTexture = baseColor?.textureSamplerValue?.texture
                let textureLoader = MTKTextureLoader(device: device)
                
                if let texURL = self.textureURL {
                    do {
                        let texture = try textureLoader.newTexture(URL: texURL, options: [.SRGB: false])
                        let index = TextureRegistry.shared.addTexture(texture, identifier: texURL.path)
                        submeshMaterial.texture_index = Int32(index)

                    } catch {
                        fatalError("Couldn't load texture: \(error)")
                    }
                }
                
                do {
                    if let cgImage = mdlTexture?.imageFromTexture() {
                        let mtlTexture = try textureLoader.newTexture(cgImage: cgImage.takeRetainedValue())
                        let identifier = "\(submesh.hashValue)\(mdlTexture.hashValue)"
                        let index = TextureRegistry.shared.addTexture(mtlTexture, identifier: identifier)
                        submeshMaterial.texture_index = Int32(index)
                    }
                } catch {
                    print("Failed to convert textures: \(error)")
                }

            } else {
                print("Failed to get material")
            }
            
            let mdlMaterial = submesh.material!

            let opacityMaterialData = extractTextureData(from: mdlMaterial, semantic: .opacity)
            let roughnessMaterialData = extractTextureData(from: mdlMaterial, semantic: .roughness)
            let metallicMaterialData = extractTextureData(from: mdlMaterial, semantic: .metallic)
            let emissionMaterialData = extractTextureData(from: mdlMaterial, semantic: .emission)
            
            for i in stride(from: 0, to: indexCount, by: 3) {
                
                let i0 = getIndexValue(from: indexData, at: i, type: indexType)
                let i1 = getIndexValue(from: indexData, at: i+1, type: indexType)
                let i2 = getIndexValue(from: indexData, at: i+2, type: indexType)
                
                var emissive = false
                var currentMaterial = submeshMaterial
                
                if let texCoordAttribute = texCoordAttribute {
                    let offset = texCoordAttribute.offset
                    
                    let t0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                        .assumingMemoryBound(to: SIMD2<Float>.self)
                    let t1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                        .assumingMemoryBound(to: SIMD2<Float>.self)
                    let t2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                        .assumingMemoryBound(to: SIMD2<Float>.self)
                    
                    let t0 = t0Ptr.pointee
                    let t1 = t1Ptr.pointee
                    let t2 = t2Ptr.pointee
                    
                    triangleTexCoords.append(t0)
                    triangleTexCoords.append(t1)
                    triangleTexCoords.append(t2)
                    
                    if let mdlMaterial = submesh.material {
                        if let opacityProperty = mdlMaterial.property(with: MDLMaterialSemantic.opacity) {
                            if let opacityMaterialData, opacityProperty.type == .texture {
                                currentMaterial.opacity = averageMaterialValue(from: opacityMaterialData, t0: t0, t1: t1, t2: t2).x
                            }
                        }
                                                
                        if let roughnessProperty = mdlMaterial.property(with: MDLMaterialSemantic.roughness) {
                            if let roughnessMaterialData, roughnessProperty.type == .texture {
                                currentMaterial.roughness_x = averageMaterialValue(from: roughnessMaterialData, t0: t0, t1: t1, t2: t2).x
                                currentMaterial.roughness_y = averageMaterialValue(from: roughnessMaterialData, t0: t0, t1: t1, t2: t2).x
                            }
                        }
                        
                        if let metallicProperty = mdlMaterial.property(with: MDLMaterialSemantic.metallic) {
                            if let metallicMaterialData, metallicProperty.type == .texture {
                                currentMaterial.metallic = averageMaterialValue(from: metallicMaterialData, t0: t0, t1: t1, t2: t2).x
                            }
                        }
                        
                        if length(self.emissionColor) > 0 {
                            emissive = true
                            lightGeometry?.lightColors.append(contentsOf: [self.emissionColor, self.emissionColor, self.emissionColor])
                        }
            
                        else if let emissionProperty = mdlMaterial.property(with: MDLMaterialSemantic.emission) {
                            if let emissionMaterialData, emissionProperty.type == .texture {
                                let emissionColor0 = sampleTexture(materialData: emissionMaterialData, at: t0)
                                let emissionColor1 = sampleTexture(materialData: emissionMaterialData, at: t1)
                                let emissionColor2 = sampleTexture(materialData: emissionMaterialData, at: t2)

                                if min(length(emissionColor0), length(emissionColor1), length(emissionColor2)) > 0.1 {
                                    emissive = true
                                    lightGeometry?.lightColors.append(contentsOf: [emissionColor0, emissionColor1, emissionColor2])
                                }
                            }
                            else if emissionProperty.type == .float3 {
                                let emissionColor = emissionProperty.float3Value
                                if length(emissionColor) > 0.1 {
                                    emissive = true
                                    lightGeometry?.lightColors.append(contentsOf: [emissionColor, emissionColor, emissionColor])
                                }
                            }
                        }
                    }

                    if emissive {
                        lightGeometry?.materials.append(currentMaterial)
                    } else {
                        triangleMaterials.append(currentMaterial)
                    }
                                        
                    if let positionAttribute = positionAttribute {
                        let offset = positionAttribute.offset
                        
                        let v0Ptr = vertexData.advanced(by: vertexStride * Int(i0) + offset)
                            .assumingMemoryBound(to: SIMD3<Float>.self)
                        let v1Ptr = vertexData.advanced(by: vertexStride * Int(i1) + offset)
                            .assumingMemoryBound(to: SIMD3<Float>.self)
                        let v2Ptr = vertexData.advanced(by: vertexStride * Int(i2) + offset)
                            .assumingMemoryBound(to: SIMD3<Float>.self)
                        
                        if emissive {
                            lightGeometry?.vertices.append(LinearAlgebra.transformPosition(position: v0Ptr.pointee, with: transform))
                            lightGeometry?.vertices.append(LinearAlgebra.transformPosition(position: v1Ptr.pointee, with: transform))
                            lightGeometry?.vertices.append(LinearAlgebra.transformPosition(position: v2Ptr.pointee, with: transform))
                        } else {
                            triangleVertices.append(LinearAlgebra.transformPosition(position: v0Ptr.pointee, with: transform))
                            triangleVertices.append(LinearAlgebra.transformPosition(position: v1Ptr.pointee, with: transform))
                            triangleVertices.append(LinearAlgebra.transformPosition(position: v2Ptr.pointee, with: transform))
                        }
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
                        
                        if emissive {
                            lightGeometry?.normals.append(LinearAlgebra.transformNormal(normal: n0, with: transform))
                            lightGeometry?.normals.append(LinearAlgebra.transformNormal(normal: n1, with: transform))
                            lightGeometry?.normals.append(LinearAlgebra.transformNormal(normal: n2, with: transform))
                        } else {
                            triangleNormals.append(LinearAlgebra.transformNormal(normal: n0, with: transform))
                            triangleNormals.append(LinearAlgebra.transformNormal(normal: n1, with: transform))
                            triangleNormals.append(LinearAlgebra.transformNormal(normal: n2, with: transform))
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
                        
                        var c0 = c0Ptr.pointee
                        var c1 = c1Ptr.pointee
                        var c2 = c2Ptr.pointee

                        if emissive {
                            lightGeometry?.colors.append(length(c0) < 1e-5 ? c0 : c0)
                            lightGeometry?.colors.append(length(c1) < 1e-5 ? c1 : c1)
                            lightGeometry?.colors.append(length(c2) < 1e-5 ? c2 : c2)
                        } else {
                            triangleColors.append(length(c0) < 1e-5 ? c0 : c0)
                            triangleColors.append(length(c1) < 1e-5 ? c1 : c1)
                            triangleColors.append(length(c2) < 1e-5 ? c2 : c2)
                        }
                    } else {
                        var color = currentColor
                        
                        if let mdlMaterial = submesh.material, let baseColorProperty = mdlMaterial.property(with: MDLMaterialSemantic.baseColor) {
                            if baseColorProperty.type == .float3 {
                                color = baseColorProperty.float3Value
                            }

                            if simd_length(color) < 1e-5 {
                                color = SIMD3<Float>(repeating: 1.0)
                            }
                        }
//                        color = .one
                        if emissive {
                            lightGeometry?.colors.append(color)
                            lightGeometry?.colors.append(color)
                            lightGeometry?.colors.append(color)
                        } else {
                            triangleColors.append(color)
                            triangleColors.append(color)
                            triangleColors.append(color)
                        }
                    }
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
        
    func extractTextureData(from material: MDLMaterial?, semantic: MDLMaterialSemantic) -> MaterialData? {
        if let material = material, let texture = material.property(with: semantic)?.textureSamplerValue?.texture, let data = texture.texelDataWithTopLeftOrigin() {
            return MaterialData(data: data,
                                width: Int(texture.dimensions.x),
                                height: Int(texture.dimensions.y))
        }
        return nil
    }
    
    func averageMaterialValue(from materialData: MaterialData, t0: SIMD2<Float>, t1: SIMD2<Float>, t2: SIMD2<Float>, readAlpha: Bool = false) -> SIMD3<Float> {
        let val0 = sampleTexture(materialData: materialData, at: t0, readAlpha: readAlpha)
        let val1 = sampleTexture(materialData: materialData, at: t1, readAlpha: readAlpha)
        let val2 = sampleTexture(materialData: materialData, at: t2, readAlpha: readAlpha)
        
        return (val0 + val1 + val2) / 3.0
    }
                                                               
    func sampleTexture(materialData: MaterialData, at uv: SIMD2<Float>, readAlpha: Bool = false) -> SIMD3<Float> {
        let data = materialData.data
        let w = materialData.width
        let h = materialData.height
        
        let u = uv.x.truncatingRemainder(dividingBy: 1.0)
        let v = uv.y.truncatingRemainder(dividingBy: 1.0)

        let px = min(max(Int((u) * Float(w)), 0), w - 1)
        let py = min(max(Int((1 - v) * Float(h)), 0), h - 1) // 1 - v is needed
        let bytesPerPixel = 4
        let rowBytes = bytesPerPixel * w
        let offset = py * rowBytes + px * bytesPerPixel
        
        if readAlpha {
            let a = Float(data[offset + 3]) / 255
            return SIMD3<Float>(a, a, a)
        }

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
    
    override func getLightGeometry() -> LightGeometry? {
        return lightGeometry
    }
}

func getVertexDescriptor() -> MDLVertexDescriptor {
    let vertexDescriptor = MDLVertexDescriptor()
    var offset = 0
    
    vertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                        format: .float3,
                                                        offset: offset,
                                                        bufferIndex: 0)
    offset += MemoryLayout<SIMD3<Float>>.stride
    
    vertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                                        format: .float3,
                                                        offset: offset,
                                                        bufferIndex: 0)
    offset += MemoryLayout<SIMD3<Float>>.stride
    
    vertexDescriptor.attributes[2] = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                                                        format: .float2,
                                                        offset: offset,
                                                        bufferIndex: 0)
    offset += MemoryLayout<SIMD2<Float>>.stride

    vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: offset)
    return vertexDescriptor
}

struct MaterialData {
    let data: Data
    let width: Int
    let height: Int
}
