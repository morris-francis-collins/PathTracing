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
    private var emissionAmplifier: Float
    
    init(device: MTLDevice, modelPath: String, defaultMaterial: Material?, defaultTexture: TextureInfo, emissionAmplifier: Float) {
        modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        self.emissionAmplifier = emissionAmplifier
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
            let material = loadMaterialWithTextures(mesh: mesh, defaultMaterial: defaultMaterial, defaultTexture: defaultTexture)
            let isEmissive = length_squared(material.emission.value) > 1e-4 || material.emission.textureIndex != -1
            let primitiveLightIndex = isEmissive ? areaLights.count : -1
            
//            if (!isEmissive) {
//                continue
//            }
            
            for i in stride(from: 0, to: Int(mesh.indexCount), by: 3) {
                let i0 = Int(mesh.indices[i + 0])
                let i1 = Int(mesh.indices[i + 1])
                let i2 = Int(mesh.indices[i + 2])
                
                vertices.append(mesh.positions[i0])
                vertices.append(mesh.positions[i1])
                vertices.append(mesh.positions[i2])
                
                normals.append(normalize(mesh.normals[i0]))
                normals.append(normalize(mesh.normals[i1]))
                normals.append(normalize(mesh.normals[i2]))
                
                texCoords.append(mesh.texCoords[i0] * defaultTexture.uvMultiplier)
                texCoords.append(mesh.texCoords[i1] * defaultTexture.uvMultiplier)
                texCoords.append(mesh.texCoords[i2] * defaultTexture.uvMultiplier)

                let materialIndex = MaterialRegistry.shared.addMaterial(material)
                
                materials.append(material)
                
                primitiveLightIndices.append(Int32(primitiveLightIndex))
                
                let primitiveData = PrimitiveData(n0: normalize(mesh.normals[i0]),
                                                  n1: normalize(mesh.normals[i1]),
                                                  n2: normalize(mesh.normals[i2]),
                                                  uv0: mesh.texCoords[i0] * defaultTexture.uvMultiplier,
                                                  uv1: mesh.texCoords[i1] * defaultTexture.uvMultiplier,
                                                  uv2: mesh.texCoords[i2] * defaultTexture.uvMultiplier,
                                                  materialIndex: Int32(materialIndex),
                                                  primitiveLightIndex: Int32(primitiveLightIndex))
                
                primitiveTriangleData.append(primitiveData)
            }

            if isEmissive {
                let averageEmission = material.emission.value // FIXME: get avg texture
                areaLights.append(AreaLight(emission: material.emission,
                                            averageEmission: averageEmission,
                                            vertices: vertices.suffix(Int(mesh.indexCount)),
                                            UVs: texCoords.suffix(Int(mesh.indexCount)))
                )
            }
        }
        
        uploadToBuffers()
    }
    
    private func loadMaterialWithTextures(mesh: MeshData, defaultMaterial: Material?, defaultTexture: TextureInfo) -> Material {
        var material = defaultMaterial ?? createStaticMaterial(
            color: SIMD3<Float>(mesh.material.color.value.x, mesh.material.color.value.y, mesh.material.color.value.z),
            refraction: mesh.material.refraction.value,
            roughness: mesh.material.roughness.value,
            metallic: mesh.material.metallic.value,
            emission: mesh.material.emission.value,
            BXDFs: mesh.material.BXDFs
        )
        
        material.emission.value *= emissionAmplifier
        
        if let textureURL = defaultTexture.textureURL {
            let index = loadTexture(device: device, textureURL: textureURL)
            material.color.textureIndex = Int32(index)
        } else if let colorIndex = loadEmbeddedTexture(mesh.embeddedColorTexture, type: "color", pixelFormat: .rgba8Unorm_srgb, sRGB: true, bytesPerPixel: 4) {
            material.color.textureIndex = Int32(colorIndex)
        }
        
        if let roughnessIndex = loadEmbeddedTexture(mesh.embeddedRoughnessTexture, type: "roughness", pixelFormat: .r8Unorm, sRGB: false, bytesPerPixel: 1) {
            material.roughness.textureIndex = Int32(roughnessIndex)
        }
        
        if let metallicIndex = loadEmbeddedTexture(mesh.embeddedMetallicTexture, type: "metallic", pixelFormat: .r8Unorm, sRGB: false, bytesPerPixel: 1) {
            material.metallic.textureIndex = Int32(metallicIndex)
        }
        
        if let emissiveIndex = loadEmbeddedTexture(mesh.embeddedEmissiveTexture, type: "emission", pixelFormat: .rgba8Unorm_srgb, sRGB: true, bytesPerPixel: 4) {
            material.emission.textureIndex = Int32(emissiveIndex)
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
    
    private func loadTexture(device: MTLDevice, textureURL: URL) -> Int32 {
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [.SRGB: false]
        
        do {
            let texture = try textureLoader.newTexture(URL: textureURL, options: options)
            let index = TextureRegistry.shared.addTexture(texture, identifier: String("defaultTexture_\(textureURL.lastPathComponent)"))
            return Int32(index)
        } catch {
            fatalError("Couldn't load texture: \(error)")
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
        descriptor.primitiveDataBuffer = primitiveTriangleDataBuffer
        descriptor.primitiveDataStride = MemoryLayout<PrimitiveData>.stride
        descriptor.primitiveDataElementSize = MemoryLayout<PrimitiveData>.size
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
