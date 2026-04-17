//
//  GameScene.swift
//  PathTracing
//
//  Created on 3/22/25.
//

import MetalKit
import simd
import CoreImage
import CoreVideo

let worldUp = SIMD3<Float>(0.0, 1.0, 0.0)
let worldRight = SIMD3<Float>(1.0, 0.0, 0.0)
let worldForward = SIMD3<Float>(0.0, 0.0, -1.0)

class GameScene: ObservableObject {
    let device: MTLDevice
    var geometries: [Geometry] = []
    var instances: [GeometryInstance] = []
        
    var lights: [Light] = []
    var lightInfo: [LightInfo] = []
    var lightTriangles: [LightTriangle] = []
    var instanceLightIndices: [Int32] = []
    
    var lightAliasEntries: [AliasEntry] = []
    var lightTriangleAliasEntries: [AliasEntry] = []
    var enviromentMapAliasEntries: [AliasEntry] = []
    
    var environmentMapTexture: MTLTexture?
    var environmentMapLightIndex: Int32 = -1

    var lightBuffer: MTLBuffer?
    var lightTriangleBuffer: MTLBuffer?
    var environmentMapCDFBuffer: MTLBuffer?
    var instanceLightIndicesBuffer: MTLBuffer?
    
    var lightAliasEntriesBuffer: MTLBuffer?
    var lightTriangleAliasEntriesBuffer: MTLBuffer?
    var environmentMapAliasEntriesBuffer: MTLBuffer?
    
    var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var cameraUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    
    var cameraSpeed: Float = 0.25
    var rotationSpeed: Float = 0.1
    
    var cameraLocations: [(SIMD3<Float>, SIMD3<Float>)] = []
    
    init(device: MTLDevice) {
        self.device = device
        TextureRegistry.shared.reset()
        MaterialRegistry.shared.reset()
        createTextCausticScene()
    }
        
    func addGeometry(_ mesh: Geometry) {
        geometries.append(mesh)
    }
    
    func addInstance(_ instance: GeometryInstance) {
        instances.append(instance)
    }
    
    func uploadToBuffers() {
        lights.removeAll() // for when we switch renderers
        lightAliasEntries.removeAll()
        lightTriangleAliasEntries.removeAll()
        
        for geometry in geometries {
            geometry.uploadToBuffers()
        }
        
        createLightAliasTable()
        createLightTriangleAliasTable()
        
        let options = getManagedBufferStorageMode()
                
        lightBuffer = device.makeBuffer(bytes: lights, length: lights.count * MemoryLayout<Light>.size, options: options)
        lightTriangleBuffer = device.makeBuffer(bytes: lightTriangles, length: lightTriangles.count * MemoryLayout<LightTriangle>.size, options: options)
        instanceLightIndicesBuffer = device.makeBuffer(bytes: instanceLightIndices, length: instanceLightIndices.count * MemoryLayout<Int32>.size, options: options)
        
        lightAliasEntriesBuffer = device.makeBuffer(bytes: lightAliasEntries, length: lightAliasEntries.count * MemoryLayout<AliasEntry>.size, options: options)
        lightTriangleAliasEntriesBuffer = device.makeBuffer(bytes: lightTriangleAliasEntries, length: lightTriangleAliasEntries.count * MemoryLayout<AliasEntry>.size, options: options)
        
        if enviromentMapAliasEntries.isEmpty { enviromentMapAliasEntries.append(AliasEntry()) }
        environmentMapAliasEntriesBuffer = device.makeBuffer(bytes: enviromentMapAliasEntries, length: enviromentMapAliasEntries.count * MemoryLayout<AliasEntry>.size, options: options)

        MaterialRegistry.shared.uploadToBuffers(device: device)
    }

    func addInstance(with geometry: Geometry, translation: SIMD3<Float> = .zero, rotation: SIMD3<Float> = .zero, scale: SIMD3<Float> = .one) {
                        
        let opaqueInstance = GeometryInstance(geometry: geometry,
                                              translation: translation,
                                              rotation: rotation,
                                              scale: scale)
        addInstance(opaqueInstance)
        
        instanceLightIndices.append(Int32(lightInfo.count))
        
        for light in geometry.areaLights {
            let (instanceLightTriangles, totalArea) = getLightTriangles(areaLight: light, transform: opaqueInstance.transform)
        
            var areaLight = Light()
            areaLight.type = AREA_LIGHT
            areaLight.index = UInt32(lightInfo.count) // BEFORE WAS INSTANCE COUNT?
            areaLight.area = AreaLight(firstTriangleIndex: UInt32(lightTriangles.count),
                                       triangleCount: UInt32(instanceLightTriangles.count),
                                       totalArea: totalArea)
            
            lightTriangles.append(contentsOf: instanceLightTriangles)
            lightInfo.append(LightInfo(light: areaLight, averageEmission: light.averageEmission))
        }
    }
    
    func calculateLuminance(_ color: SIMD3<Float>) -> Float {
        return dot(SIMD3<Float>(0.2126, 0.7152, 0.0722), color)
    }
    
    func createAliasTable(weights: [Float]) -> [AliasEntry] {
        let n = weights.count
        guard n > 0 else { fatalError("Alias table weights empty.") }
        let totalWeight = weights.reduce(0.0, +)
        
        var PMFs: [Float] = []
        var U: [Float] = []

        for i in 0..<n {
            PMFs.append(weights[i] / totalWeight)
            U.append(weights[i] * Float(n) / totalWeight)
        }
        
        var small: [Int] = []
        var large: [Int] = []
        
        for i in 0..<n {
            if U[i] <= 1.0 {
                small.append(i)
            } else {
                large.append(i)
            }
        }
        
        var aliasTable = [AliasEntry](repeating: AliasEntry(), count: n)
        
        while !small.isEmpty && !large.isEmpty {
            let s = small.removeLast()
            let l = large.removeLast()
            
            aliasTable[s] = AliasEntry(acceptanceProbability: U[s], alias: UInt32(l), PMF: PMFs[s])
            U[l] -= 1.0 - U[s]
                
            if U[l] <= 1.0 {
                small.append(l)
            } else {
                large.append(l)
            }
        }
        
        for i in small {
            aliasTable[i] = AliasEntry(acceptanceProbability: 1.0, alias: UInt32(i), PMF: PMFs[i])
        }
        
        for i in large {
            aliasTable[i] = AliasEntry(acceptanceProbability: 1.0, alias: UInt32(i), PMF: PMFs[i])
        }
                
        return aliasTable
    }
    
    func createLightAliasTable() {
        func getLightPower(_ lightInfo: LightInfo) -> Float {
            switch (lightInfo.light.type) {
            case POINT_LIGHT: return 4.0 * .pi * calculateLuminance(lightInfo.averageEmission);
            case AREA_LIGHT: return lightInfo.light.area.totalArea * calculateLuminance(lightInfo.averageEmission);
            case DIRECTIONAL_LIGHT: return .pi * SCENE_RADIUS * SCENE_RADIUS * calculateLuminance(lightInfo.averageEmission)
            case ENVIRONMENT_MAP: return 4.0 * .pi * .pi * SCENE_RADIUS * SCENE_RADIUS * calculateLuminance(lightInfo.averageEmission)
            default: fatalError("Unknown light type.")
            }
        }

        var powers: [Float] = []
        
        for light in lightInfo {
            powers.append(getLightPower(light))
            lights.append(light.light)
        }
        
        lightAliasEntries = createAliasTable(weights: powers)
    }
    
    func createLightTriangleAliasTable() {
        if lightTriangles.isEmpty {
            lightTriangles = [LightTriangle](repeating: LightTriangle(), count: 1)
            lightTriangleAliasEntries = [AliasEntry](repeating: AliasEntry(), count: 1)
            return
        }
        
        lightTriangleAliasEntries = [AliasEntry](repeating: AliasEntry(), count: lightTriangles.count)

        for light in lightInfo {
            if light.light.type != AREA_LIGHT {
                continue
            }
            
            let first = Int(light.light.area.firstTriangleIndex)
            let count = Int(light.light.area.triangleCount)
            
            var weights: [Float] = []
            
            for i in first..<(first + count) {
                let triangle = lightTriangles[i]
                let area = 0.5 * length(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0))
                weights.append(area)
            }

            let localTable = createAliasTable(weights: weights)

            for i in 0..<count {
                var entry = localTable[i]
                entry.alias += UInt32(first) // shift local to global
                lightTriangleAliasEntries[first + i] = entry
            }
        }
    }
        
    func getLightTriangles(areaLight: AreaLightData, transform: simd_float4x4) -> ([LightTriangle], Float) {
        let vertices = areaLight.vertices
        let UVs = areaLight.UVs
        var instanceLightTriangles: [LightTriangle] = []
        var totalArea: Float = 0

        for i in 0..<vertices.count / 3 {
            let v0World = transform * SIMD4<Float>(vertices[3 * i + 0], 1.0)
            let v1World = transform * SIMD4<Float>(vertices[3 * i + 1], 1.0)
            let v2World = transform * SIMD4<Float>(vertices[3 * i + 2], 1.0)
            
            let v0 = SIMD3<Float>(v0World.x, v0World.y, v0World.z)
            let v1 = SIMD3<Float>(v1World.x, v1World.y, v1World.z)
            let v2 = SIMD3<Float>(v2World.x, v2World.y, v2World.z)
            
            let area = 0.5 * length(cross(v1 - v0, v2 - v0))
            totalArea += area
            
            instanceLightTriangles.append(LightTriangle(v0: v0, v1: v1, v2: v2,
                                                        uv0: UVs[3 * i + 0], uv1: UVs[3 * i + 1], uv2: UVs[3 * i + 2],
                                                        emission: areaLight.emission,
                                                        emissionTextureIndex: areaLight.emissionTextureIndex))
        }
        
        return (instanceLightTriangles, totalArea)
    }
            
    func addPointLight(position: SIMD3<Float>, color: SIMD3<Float>) {
        var pointLight = Light()
        pointLight.type = POINT_LIGHT
        pointLight.index = UInt32(lights.count)
        pointLight.point = PointLight(position: position, color: color)
        
        lightInfo.append(LightInfo(light: pointLight, averageEmission: color))
    }
    
    func addDirectionalLight(direction: SIMD3<Float>, color: SIMD3<Float>) {
        var directionalLight = Light()
        directionalLight.type = DIRECTIONAL_LIGHT
        directionalLight.index = UInt32(lights.count)
        directionalLight.directional = DirectionalLight(direction: direction, color: color)
        
        lightInfo.append(LightInfo(light: directionalLight, averageEmission: color))
    }
    
    func addEnvironmentMap(textureURL: URL, emissionAmplifier: Float = 1.0) { // TODO: add texture merging if we want to overlay multiple env maps
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [.SRGB: false, .textureStorageMode: MTLStorageMode.shared.rawValue]
                
        do {
            environmentMapTexture = try textureLoader.newTexture(URL: textureURL, options: options)
            environmentMapLightIndex = Int32(lightInfo.count)
        } catch {
            fatalError("Couldn't load environemntMap texture texture: \(error)")
        }
                
        let width = environmentMapTexture!.width
        let height = environmentMapTexture!.height
        var pixels = [Float](repeating: 0, count: width * height * 4)
        
        environmentMapTexture!.getBytes(&pixels,
                                        bytesPerRow: width * MemoryLayout<Float>.size * 4,
                                        from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1)),
                                        mipmapLevel: 0)
        
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i + 0] *= emissionAmplifier
            pixels[i + 1] *= emissionAmplifier
            pixels[i + 2] *= emissionAmplifier
        }
                
        var weights = [Float](repeating: 0, count: width * height)
        var totalColor: SIMD3<Float> = .zero
        
        for r in 0..<height {
            for c in 0..<width {
                let idx = (r * width + c) * 4
                let color = SIMD3<Float>(pixels[idx + 0], pixels[idx + 1], pixels[idx + 2])
                totalColor += color
                                
                let v = (Float(r) + 0.5) / Float(height)
                let theta = v * .pi
                let sinTheta = sin(theta)

                let luminance = calculateLuminance(color) * sinTheta
                weights[r * width + c] = luminance
            }
        }
        
        environmentMapTexture!.replace(
            region: MTLRegion(origin: .init(), size: MTLSize(width: width, height: height, depth: 1)),
            mipmapLevel: 0,
            withBytes: pixels,
            bytesPerRow: width * MemoryLayout<Float>.size * 4
        )
        
        var light = Light()
        light.type = ENVIRONMENT_MAP
        light.index = UInt32(lightInfo.count)
        light.environment = EnvironmentMap(width: UInt32(width),
                                           height: UInt32(height))
        
        lightInfo.append(LightInfo(light: light, averageEmission: totalColor / Float(width * height)))
        
        enviromentMapAliasEntries.append(contentsOf: createAliasTable(weights: weights))
    }

    func addAssimpGeometry(fileName: String, fileExtension: String, defaultMaterial: Material? = nil, defaultTexture: TextureInfo = TextureInfo(), emissionAmplifier: Float = 1.0) -> AssimpGeometry {
        guard let modelPath = Bundle.main.path(forResource: fileName, ofType: fileExtension) else {
            fatalError("[GameScene] Failed to find resource \(fileName).\(fileExtension)")
        }

        let geometry = AssimpGeometry(device: device,
                                      modelPath: modelPath,
                                      defaultMaterial: defaultMaterial,
                                      defaultTexture: defaultTexture,
                                      emissionAmplifier: emissionAmplifier)
        addGeometry(geometry)
        return geometry
    }
    
    func addGLTFGeometry(fileName: String, fileExtension: String, defaultMaterial: Material? = nil, emissionAmplifier: Float = 1.0) -> GLTFGeometry {
        guard let modelPath = Bundle.main.path(forResource: fileName, ofType: fileExtension) else {
            fatalError("[GameScene] Failed to find resource \(fileName).\(fileExtension)")
        }
 
        let geometry = GLTFGeometry(device: device,
                                    modelPath: modelPath,
                                    emissionAmplifier: emissionAmplifier)
        addGeometry(geometry)
        return geometry
    }
}

struct LightInfo {
    var light: Light
    var averageEmission: SIMD3<Float>
}
