//
//  GLTFGeometry.swift
//  PathTracing
//
//  Created on 3/22/25.
//

import MetalKit
import Foundation
import simd

class GLTFGeometry: Geometry {
    private var modelName: String

    init(device: MTLDevice, modelPath: String, emissionAmplifier: Float = 1.0) {
        modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        super.init(device: device)
        lightGeometry = LightGeometry(device: device)

        guard let sceneData = loadGLTFScene(modelPath.cString(using: .utf8)) else {
            print("[GLTFGeometry] Failed to load: \(modelPath)")
            return
        }
        defer { freeGLTFScene(sceneData) }

        let scene = sceneData.pointee

        let textureMap = uploadImages(scene: scene, device: device)

        let gpuMaterials = convertMaterials(scene: scene, textureMap: textureMap, emissionAmplifier: emissionAmplifier)

        for p in 0..<Int(scene.primitiveCount) {
            let prim = scene.primitives[p]
            let matIdx = min(Int(prim.materialIndex), gpuMaterials.count - 1)
            let material = gpuMaterials[matIdx]

            let emissionMag = length_squared(material.emissionValue * material.emissiveStrength)
            let isEmissive = emissionMag > 1e-4 || material.emissionTextureIndex != -1
            let primitiveLightIndex = isEmissive ? areaLights.count : -1

            let vCount = Int(prim.vertexCount)
            let triCount = vCount / 3
            let materialIndex = MaterialRegistry.shared.addMaterial(material)

            for tri in 0..<triCount {
                let baseIdx = tri * 3
                let v0 = prim.vertices[baseIdx + 0]
                let v1 = prim.vertices[baseIdx + 1]
                let v2 = prim.vertices[baseIdx + 2]

                vertices.append(v0.position)
                vertices.append(v1.position)
                vertices.append(v2.position)

                normals.append(v0.normal)
                normals.append(v1.normal)
                normals.append(v2.normal)

                texCoords.append(v0.texCoord)
                texCoords.append(v1.texCoord)
                texCoords.append(v2.texCoord)

                materials.append(material)

                primitiveLightIndices.append(Int32(primitiveLightIndex))

                let primitiveData = PrimitiveData(
                    n0: v0.normal,
                    n1: v1.normal,
                    n2: v2.normal,
                    uv0: v0.texCoord,
                    uv1: v1.texCoord,
                    uv2: v2.texCoord,
                    materialIndex: Int32(materialIndex),
                    primitiveLightIndex: Int32(primitiveLightIndex)
                )
                primitiveTriangleData.append(primitiveData)
            }

            if isEmissive {
                let emissionVerts = Array(vertices.suffix(vCount))
                let emissionUVs = Array(texCoords.suffix(vCount))
                let avgEmission = material.emissionValue * material.emissiveStrength
                areaLights.append(AreaLight(
                    emission: material.emissionValue,
                    emissionTextureIndex: material.emissionTextureIndex,
                    averageEmission: avgEmission,
                    vertices: emissionVerts,
                    UVs: emissionUVs
                ))
            }
        }

        print("[GLTFGeometry] \(modelName): \(vertices.count / 3) triangles, "
              + "\(areaLights.count) area lights")

        uploadToBuffers()
    }

    private func uploadImages(scene: GLTFSceneData, device: MTLDevice) -> [Int : Int] {
        var map: [Int : Int] = [:]
        let textureLoader = MTKTextureLoader(device: device)

        for i in 0..<Int(scene.imageCount) {
            let img = scene.images[i]
            guard img.data != nil, img.width > 0, img.height > 0 else { continue }

            let isSRGB = (img.usage == GLTFImageUsage(rawValue: 0) || img.usage == GLTFImageUsage(rawValue: 3))

            let pixelFormat: MTLPixelFormat = isSRGB ? .rgba8Unorm_srgb : .rgba8Unorm

            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: pixelFormat,
                width: Int(img.width),
                height: Int(img.height),
                mipmapped: true
            )
            
            desc.usage = [.shaderRead]

            guard let texture = device.makeTexture(descriptor: desc) else { continue }

            texture.replace(
                region: MTLRegionMake2D(0, 0, Int(img.width), Int(img.height)),
                mipmapLevel: 0,
                withBytes: img.data,
                bytesPerRow: Int(img.width) * 4
            )

            let identifier = "gltf_\(modelName)_img\(i)"
            let idx = TextureRegistry.shared.addTexture(texture, identifier: identifier)
            map[i] = idx
        }

        return map
    }

    private func convertMaterials(scene: GLTFSceneData, textureMap: [Int: Int], emissionAmplifier: Float) -> [Material] {
        var result: [Material] = []

        for i in 0..<Int(scene.materialCount) {
            let src = scene.materials[i]

            func remapTex(_ loaderIndex: Int32) -> Int32 {
                if loaderIndex < 0 { return -1 }
                return Int32(textureMap[Int(loaderIndex)] ?? -1)
            }

            var mat = src
            mat.emissiveStrength *= emissionAmplifier
            
            result.append(mat)
        }

        return result
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
}
