//
//  AssimpGeometry.swift
//  PathTracing
//
//  Created on 9/18/25.
//

import MetalKit
import Foundation
import simd

class AssimpGeometry: Geometry {
    private var sceneData: UnsafeMutablePointer<SceneData>?
    private var textureLoader: MTKTextureLoader?
    private var modelName: String
    
    init(device: MTLDevice, modelPath: String, defaultMaterial: Material? = nil) {
        modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        super.init(device: device)
        textureLoader = MTKTextureLoader(device: device)
        lightGeometry = LightGeometry(device: device)
        
        sceneData = loadModel(modelPath.cString(using: .utf8))
        
        guard let scene = sceneData?.pointee else {
            print("Failed to load model from: \(modelPath)")
            return
        }
        
        print("Processing \(scene.meshCount) meshes")
        
        for m in 0..<Int(scene.meshCount) {
            let mesh = scene.meshes[m]
            let material = loadMaterialWithTextures(mesh: mesh, defaultMaterial: defaultMaterial)
    
            for i in stride(from: 0, to: Int(mesh.indexCount), by: 3) {
                let i0 = Int(mesh.indices[i + 0])
                let i1 = Int(mesh.indices[i + 1])
                let i2 = Int(mesh.indices[i + 2])
                
                vertices.append(SIMD3<Float>(mesh.positions[i0].x, mesh.positions[i0].y, mesh.positions[i0].z))
                vertices.append(SIMD3<Float>(mesh.positions[i1].x, mesh.positions[i1].y, mesh.positions[i1].z))
                vertices.append(SIMD3<Float>(mesh.positions[i2].x, mesh.positions[i2].y, mesh.positions[i2].z))
                
                normals.append(normalize(SIMD3<Float>(mesh.normals[i0].x, mesh.normals[i0].y, mesh.normals[i0].z)))
                normals.append(normalize(SIMD3<Float>(mesh.normals[i1].x, mesh.normals[i1].y, mesh.normals[i1].z)))
                normals.append(normalize(SIMD3<Float>(mesh.normals[i2].x, mesh.normals[i2].y, mesh.normals[i2].z)))
                
                texCoords.append(SIMD2<Float>(mesh.texCoords[i0].x, mesh.texCoords[i0].y))
                texCoords.append(SIMD2<Float>(mesh.texCoords[i1].x, mesh.texCoords[i1].y))
                texCoords.append(SIMD2<Float>(mesh.texCoords[i2].x, mesh.texCoords[i2].y))
                
                materials.append(material)

                colors.append(contentsOf: [material.color.value,
                                          material.color.value,
                                          material.color.value])
            }
        }
        
        uploadToBuffers()
    }
    
    private func loadMaterialWithTextures(mesh: MeshData, defaultMaterial: Material?) -> Material {
        var material = defaultMaterial ?? createStaticMaterial(
            color: SIMD3<Float>(mesh.material.color.value.x, mesh.material.color.value.y, mesh.material.color.value.z),
            refraction: mesh.material.refraction.value,
            roughness: mesh.material.roughness.value,
            metallic: mesh.material.metallic.value,
            BXDFs: mesh.material.BXDFs
        )
        
        if let colorIndex = loadEmbeddedTexture(mesh.embeddedColorTexture, type: "color", pixelFormat: .rgba8Unorm_srgb, sRGB: true, bytesPerPixel: 4) {
            material.color.textureIndex = Int32(colorIndex)
        }
        
        if let roughnessIndex = loadEmbeddedTexture(mesh.embeddedRoughnessTexture, type: "roughness", pixelFormat: .r8Unorm, sRGB: false, bytesPerPixel: 1) {
            material.roughness.textureIndex = Int32(roughnessIndex)
        }
        
        if let metallicIndex = loadEmbeddedTexture(mesh.embeddedMetallicTexture, type: "metallic", pixelFormat: .r8Unorm, sRGB: false, bytesPerPixel: 1) {
            material.metallic.textureIndex = Int32(metallicIndex)
        }
        
        return material
    }
        
    private func loadEmbeddedTexture(_ embeddedPtr: UnsafeMutablePointer<EmbeddedTexture>?, type: String, pixelFormat: MTLPixelFormat, sRGB: Bool, bytesPerPixel: Int) -> Int? {
        guard let embeddedPtr = embeddedPtr else { return nil }
        
        let embedded = embeddedPtr.pointee
        let data = Data(bytes: embedded.data, count: Int(embedded.dataSize))
        
        if embedded.isCompressed {
            guard let texture = try? textureLoader!.newTexture(data: data, options: [.SRGB: NSNumber(value: sRGB), .generateMipmaps: NSNumber(value: true)])
            else { return nil }
            
            let index = TextureRegistry.shared.addTexture(
                texture,
                identifier: "Embedded_\(modelName)_\(type)_\(embedded.index)"
            )
            print("Loaded embedded \(type) texture at index \(index)")
            return index
            
        } else {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: pixelFormat,
                width: Int(embedded.width),
                height: Int(embedded.height),
                mipmapped: true
            )
            
            guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
            
            texture.replace(
                region: MTLRegionMake2D(0, 0, Int(embedded.width), Int(embedded.height)),
                mipmapLevel: 0,
                withBytes: embedded.data,
                bytesPerRow: Int(embedded.width) * bytesPerPixel
            )
            
            let index = TextureRegistry.shared.addTexture(
                texture,
                identifier: "EmbeddedRaw_\(modelName)_\(type)_\(embedded.index)"
            )
            print("Loaded embedded raw \(type) texture at index \(index)")
            return index
        }
    }
    
    deinit {
        if let scene = sceneData {
            freeSceneData(scene)
        }
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
