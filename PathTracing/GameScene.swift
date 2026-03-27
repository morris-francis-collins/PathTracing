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

let GEOMETRY_MASK_TRIANGLE: UInt32 = 1 << 0
let GEOMETRY_MASK_SPHERE: UInt32 = 1 << 1
let GEOMETRY_MASK_LIGHT: UInt32 = 1 << 2
let GEOMETRY_MASK_TRANSPARENT: UInt32 = 1 << 3
let GEOMETRY_MASK_OPAQUE: UInt32 = 1 << 4

let worldUp = SIMD3<Float>(0.0, 1.0, 0.0)
let worldRight = SIMD3<Float>(1.0, 0.0, 0.0)
let worldForward = SIMD3<Float>(0.0, 0.0, -1.0)

class GameScene: ObservableObject {
    let device: MTLDevice
    var geometries: [Geometry] = []
    var instances: [GeometryInstance] = []
    
    var lights: [Light] = []
    var lightTriangles: [LightTriangle] = []
    var instanceLightIndices: [Int32] = []
    var environmentMapCDF: [Float] = []
    var environmentMapTexture: MTLTexture?

    var lightBuffer: MTLBuffer?
    var lightTriangleBuffer: MTLBuffer?
    var environmentMapCDFBuffer: MTLBuffer?
    var instanceLightIndicesBuffer: MTLBuffer?
    
    var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var cameraUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    
    var cameraSpeed: Float = 0.25
    var rotationSpeed: Float = 0.1
    
    var cameraLocations: [(SIMD3<Float>, SIMD3<Float>)] = []
    
    init(device: MTLDevice) {
        self.device = device
        createTheWhiteRoom()
    }
        
    func addGeometry(_ mesh: Geometry) {
        geometries.append(mesh)
    }
    
    func addInstance(_ instance: GeometryInstance) {
        instances.append(instance)
    }
    
    func uploadToBuffers() {
        for geometry in geometries {
            geometry.uploadToBuffers()
        }
        
        let options = getManagedBufferStorageMode()

        if (environmentMapTexture == nil) { // FIXME: placeholder texture returns nan values
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float,
                width:  1,
                height: 1,
                mipmapped: false
            )
            desc.usage = .shaderRead
            let tex = device.makeTexture(descriptor: desc)!
            var black: [Float32] = [0, 0, 0, 0]
            tex.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: 1, height: 1, depth: 1)),
                        mipmapLevel: 0,
                        withBytes: &black,
                        bytesPerRow: 16)
            
            environmentMapTexture = tex
            environmentMapCDF = [1]

            let newEnvironmentLight = Light(type: ENVIRONMENT_MAP,
                                            index: UInt32(lights.count),
                                            delta: false,
                                            position: .zero,
                                            color: .zero,
                                            firstTriangleIndex: 0,
                                            triangleCount: 0,
                                            totalArea: 0,
                                            direction: .zero)

            lights.append(newEnvironmentLight)
        }
        
        if (lightTriangles.isEmpty) {
            lightTriangles.append(LightTriangle(v0: .zero, v1: .zero, v2: .zero, uv0: .zero, uv1: .zero, uv2: .zero, emission: .zero, emissionTextureIndex: -1, CDF: 0.0))
        }
        
        print("instanceLightIndices", instanceLightIndices)

        lightBuffer = device.makeBuffer(bytes: lights, length: lights.count * MemoryLayout<Light>.size, options: options)
        lightTriangleBuffer = device.makeBuffer(bytes: lightTriangles, length: lightTriangles.count * MemoryLayout<LightTriangle>.size, options: options)
        environmentMapCDFBuffer = device.makeBuffer(bytes: environmentMapCDF, length: environmentMapCDF.count * MemoryLayout<Float>.size, options: options)
        instanceLightIndicesBuffer = device.makeBuffer(bytes: instanceLightIndices, length: instanceLightIndices.count * MemoryLayout<Int32>.size, options: options)
        
        MaterialRegistry.shared.uploadToBuffers(device: device)
    }
    
    func buildBox(width: Float = 6.0, height: Float = 4.0, depth: Float = 6.0) {
        let epsilon: Float = 1e-3
        let wallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: PLASTIC)
        
        // floor
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // left wall
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(-width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // right wall
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // back wall
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // ceiling
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // front wall
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
    }

    func buildColorfulBox(width: Float = 4.0, height: Float = 3.6, depth: Float = 4.0) {
        let wallThickness: Float = 0.4
        
        let whiteWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                                  defaultMaterial: PLASTIC)
        let redWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                                defaultMaterial: colorMaterial(material: PLASTIC, color: 0.7 * RED))
        let greenWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                                  defaultMaterial: colorMaterial(material: PLASTIC, color: 0.7 * GREEN))
        let checkeredWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                                      defaultMaterial: PLASTIC,
                                                      defaultTexture: TextureInfo(textureURL: checkerBoardURL, uvMultiplier: 5.0))
        
        // floor
        addInstance(
            with: whiteWallGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, wallThickness, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // back wall
        addInstance(
            with: checkeredWallGeometry,
            translation: SIMD3<Float>(0.0, 1.8, 2.2),
            rotation: SIMD3<Float>(0, 0, .pi / 2),
            scale: SIMD3<Float>(4.0, 4.0, wallThickness),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // left wall
        addInstance(
            with: redWallGeometry,
            translation: SIMD3<Float>(2.2, 1.8, 0.0),
            rotation: SIMD3<Float>(0, .pi / 2, .pi / 2),
            scale: SIMD3<Float>(4.0, 4.0, wallThickness),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // right wall
        addInstance(
            with: greenWallGeometry,
            translation: SIMD3<Float>(-2.2, 1.8, 0.0),
            rotation: SIMD3<Float>(0, .pi / 2, .pi / 2),
            scale: SIMD3<Float>(4.0, 4.0, wallThickness),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // ceiling
        addInstance(
            with: checkeredWallGeometry,
            translation: SIMD3<Float>(0, height, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, wallThickness, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
    }
    
    func buildSegmentedBox(extra: Bool = false) {
        let width: Float = 8.0
        let height: Float = 4.0
        let depth: Float = 8.0
        let epsilon: Float = 1e-3
        
        let wallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: PLASTIC)
        let glassWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: GLASS)
        let mirrorCubeGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: MIRROR)
        let checkeredWallGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: PLASTIC,
                                                     defaultTexture: TextureInfo(textureURL: checkerBoardURL, uvMultiplier: 5.0))

        // floor
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, 0.0, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, epsilon, depth),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // left wall
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(-width / 2, height / 2, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(epsilon, height, depth),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // right wall
        addInstance(with: checkeredWallGeometry,
                    translation: SIMD3<Float>(width / 2, height / 2, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(epsilon, height, depth),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // back wall
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, height, epsilon),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // front wall
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, height / 2, depth / 2),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, height, epsilon),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // ceiling
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, height, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, epsilon, depth),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // separator
        addInstance(with: glassWallGeometry,
                    translation: SIMD3<Float>(1.0, height / 2, -1.5),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(100 * epsilon, height, 5 * depth / 8),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // right mirror
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(2.5, 2.0, -3.9),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(2.0, 4.0, 0.01),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        // left mirror
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(-3.0, 2.0, -3.9),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(1.0, 3.0, 0.01),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        if (!extra) { return }
        
        // wall mirrors
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(0.9, 2, 0),
                    scale: SIMD3<Float>(epsilon, 1, 1),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(1.1, 2, -2.25),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(epsilon, 4, 1.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(1.1, 2, -0.25),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(epsilon, 4, 1.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
    }
        
    func buildCornellBox(width: Float, height: Float, depth: Float) {
        let epsilon: Float = 1e-3

        let whiteCubeGeometry = addAssimpGeometry(fileName: "cube",
                                                  fileExtension: "obj",
                                                  defaultMaterial: colorMaterial(material: PLASTIC, color: 0.7 * .one)
        )
        
        let redCubeGeometry = addAssimpGeometry(fileName: "cube",
                                                  fileExtension: "obj",
                                                  defaultMaterial: colorMaterial(material: PLASTIC, color: 0.7 * RED)
        )

        let greenCubeGeometry = addAssimpGeometry(fileName: "cube",
                                                  fileExtension: "obj",
                                                  defaultMaterial: colorMaterial(material: PLASTIC, color: 0.7 * GREEN)
        )
        
        addInstance( // floor
            with: whiteCubeGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth)
        )
        
        addInstance( // left wall
            with: redCubeGeometry,
            translation: SIMD3<Float>(-width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth)
        )
        
        addInstance( // right wall
            with: greenCubeGeometry,
            translation: SIMD3<Float>(width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth)
        )
        
        addInstance( // back wall
            with: whiteCubeGeometry,
            translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon)
        )
        
        addInstance( // ceiling
            with: whiteCubeGeometry,
            translation: SIMD3<Float>(0.0, height, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth)
        )
        
        addInstance( // left box
            with: whiteCubeGeometry,
            translation: SIMD3<Float>(-1.2, 0.75, 0.5),
            rotation: SIMD3<Float>(0, -0.7, 0),
            scale: SIMD3<Float>(1.5, 1.5, 1.5)
        )
        
        addInstance( // right box
            with: whiteCubeGeometry,
            translation: SIMD3<Float>(1.65, 1.25, -0.1),
            rotation: SIMD3<Float>(0, 1, 0),
            scale: SIMD3<Float>(1.35, 2.5, 1.35)
        )
    }
    
    func createBasicScene() {
        cameraLocations = [
            (SIMD3<Float>(-2.5, 1.5, 2.5), SIMD3<Float>(0.0, 2.0, 0.0)),
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildBox()
                
        addPointLight(position: SIMD3<Float>(0, 2, 0), color: 10 * .one)
    }
        
    func createColorfulDragonScene() {
        cameraLocations = [
            (SIMD3<Float>(0.0, 2.0, -5.0), SIMD3<Float>(0.0, 1.75, 0.0)),
            (SIMD3<Float>(1.5330207, 0.7450087, 1.7244401), SIMD3<Float>(-2.7430966, 2.4618351, -0.23255682))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildColorfulBox()
                
        let goldMaterial = createStaticMaterial(color: SIMD3<Float>(1.0, 0.84, 0.6), roughness: 0.35, metallic: 1.0, emission: .zero)
        let dragonGeometry = addAssimpGeometry(fileName: "stanford_dragon", fileExtension: "obj", defaultMaterial: goldMaterial)
        
        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(0.0, -0.5, 0.0),
                    rotation: SIMD3<Float>(0, .pi + 0.5, 0),
                    scale: SIMD3<Float>(0.15, 0.15, 0.15)
        )
        
        let lightGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                              defaultMaterial: createEmissiveMaterial(color: SIMD3<Float>(1.0, 0.8, 0.4)), emissionAmplifier: 10.0)

        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.35, -1.0),
                    scale: SIMD3<Float>(3.99, 0.01, 0.8))
    
        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.35, 1.0),
                    scale: SIMD3<Float>(3.99, 0.01, 0.8))
                
//        addEnvironmentMap(textureURL: skyURL)

//        addPointLight(position: SIMD3<Float>(-1.0, 2.5, 0.0), color: 10 * SIMD3<Float>(0.0, 0, 1.0))
        
//        addDirectionalLight(direction: SIMD3<Float>(0.0, 0.0, 1.0), color: 10 * .one)
    }
    
    func createLivelyScene() {
        cameraLocations = [
            (SIMD3<Float>(1.208283, 2.3361523, 3.7958465), SIMD3<Float>(0.20828247, 1.836154, -2.2041554)),
            (SIMD3<Float>(0.4791577, 2.3361523, 3.9679723), SIMD3<Float>(0.08315414, 1.836154, -2.1018896)),
            (SIMD3<Float>(3.8662832, 2.6109245, 1.2534332), SIMD3<Float>(-1.6490605, 1.5061607, -1.1152222)),
            (SIMD3<Float>(-3.773082, 2.496683, 3.7538624), SIMD3<Float>(-0.21327114, 1.3919188, -1.0790801)),
            (SIMD3<Float>(1.6659794, 2.5656717, 2.5221448), SIMD3<Float>(2.4729695, 1.4609078, -3.4258208)),
            (SIMD3<Float>(-3.7167082, 3.2056031, -3.702989), SIMD3<Float>(0.89975977, 1.5071131, -0.09003043)),
            (SIMD3<Float>(-7.706562, 4.6435676, 2.1494431), SIMD3<Float>(-2.7667365, 1.8142903, -0.051392317))
        ]

        (cameraPosition, cameraTarget) = cameraLocations[2]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildSegmentedBox()
        
        let tableGeometry = addAssimpGeometry(fileName: "Industrial_Table", fileExtension: "glb", defaultMaterial: PLASTIC)
        let glassBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: GLASS)
        let mirrorBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: MIRROR)
        let mirrorGeometry = addAssimpGeometry(fileName: "stand_mirror", fileExtension: "glb", defaultMaterial: MIRROR)
        let teaTableGeometry = addAssimpGeometry(fileName: "tea_table", fileExtension: "glb", defaultMaterial: PLASTIC)
        let couchGeometry = addAssimpGeometry(fileName: "basic_couch", fileExtension: "glb", defaultMaterial: PLASTIC)
        let hangingLightGeometry = addAssimpGeometry(fileName: "hanging_light", fileExtension: "glb", emissionAmplifier: 20)
        let floorLampGeometry = addAssimpGeometry(fileName: "floor_lamp", fileExtension: "glb", emissionAmplifier: 7.5)
        let lightBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj",
                                                  defaultMaterial: createEmissiveMaterial(color: .one), emissionAmplifier: 50)

        // table
        addInstance(with: tableGeometry,
                    translation: SIMD3<Float>(2.5, 0.0, -1.5),
                    rotation: SIMD3<Float>(0, .pi/2, 0),
                    scale: 0.02 * SIMD3<Float>(1.5, 1, 1.25),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // glass ball on table
        addInstance(with: glassBallGeometry,
                    translation: SIMD3<Float>(2.5, 1.5 + 0.375, -0.3),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.75, 0.75, 0.75),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // floor glass ball
        addInstance(with: glassBallGeometry,
                    translation: SIMD3<Float>(0.0, 0.5, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // mirror ball
        addInstance(with: mirrorBallGeometry,
                    translation: SIMD3<Float>(-1.5, 1.1, -0.75),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // standing mirror
        addInstance(with: mirrorGeometry,
                    translation: SIMD3<Float>(-0.5, 1.75, -2),
                    rotation: SIMD3<Float>(0, -.pi/8, 0),
                    scale: 30 * SIMD3<Float>(0.02, 0.012, 0.01),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        // tea table
        addInstance(with: teaTableGeometry,
                    translation: SIMD3<Float>(-1.5, 0, 0.0),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(2, 2, 2),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        // couch
        addInstance(with: couchGeometry,
                    translation: SIMD3<Float>(-3.15, 0, 0),
                    rotation: SIMD3<Float>(0, .pi/2, 0),
                    scale: SIMD3<Float>(0.017, 0.017, 0.017),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        // hanging light
        addInstance(with: hangingLightGeometry,
                    translation: SIMD3<Float>(-0.5, -0.25, 2.0),
                    rotation: SIMD3<Float>(0, .pi/2, 0),
                    scale: SIMD3<Float>(0.005, 0.005, 0.005),
                    mask: GEOMETRY_MASK_OPAQUE)
        
        // floor lamp
        addInstance(with: floorLampGeometry,
                    translation: SIMD3<Float>(-3.5, 0.0, -3.5),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.015, 0.015, 0.015),
                    mask: GEOMETRY_MASK_OPAQUE)
        
        // light balls
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(-0.5, 1.75, -2.75),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(2.5, 1.5 + 0.375 + 1.5, -1.0),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5))
    }
    
    func createDifficultScene() {
        cameraPosition = SIMD3<Float>(0, 2.5, 6)
        cameraTarget = SIMD3<Float>(0, 2, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        cameraLocations = [
            (cameraPosition, cameraTarget),
            (SIMD3<Float>(2.9280746, 2.5814812, 3.0649862), SIMD3<Float>(-0.489712, 1.8794974, -1.8417473)),
            (SIMD3<Float>(3.860585, 2.7261002, -0.094045304), SIMD3<Float>(-0.15811086, -0.75679064, -2.9171062)),
            (SIMD3<Float>(3.860585, 1.2261002, 0.9059547), SIMD3<Float>(-0.120718956, -1.2116656, -2.896234)),
            (SIMD3<Float>(-1.139415, 2.2261002, -0.094045304), SIMD3<Float>(4.7482796, 1.22613, -0.85901386))
        ]
        
        buildBox(width: 8.0, height: 4.0, depth: 8.0)

        let plasticBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: PLASTIC)
        let glassBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: GLASS)
        let mirrorCubeGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: MIRROR)
        let ringGeometry = addAssimpGeometry(fileName: "ring", fileExtension: "obj", defaultMaterial: MIRROR)
        let glassCubeGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: GLASS)
        let torusGeometry = addAssimpGeometry(fileName: "torus", fileExtension: "obj", defaultMaterial: PLASTIC)
        let lightBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj",
                                                   defaultMaterial: createEmissiveMaterial(color: .one),
                                                   emissionAmplifier: 5)
        
        // Ring on floor
        addInstance(with: ringGeometry,
                    translation: SIMD3<Float>(-2, 0.25, 0),
                    scale: SIMD3<Float>(2.0, 0.5, 2.0),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // Ring on wall
        addInstance(with: ringGeometry,
                    translation: SIMD3<Float>(-3.75, 2, 0),
                    rotation: SIMD3<Float>(0, 0, .pi/2),
                    scale: SIMD3<Float>(2.0, 0.5, 2.0),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // Glass cube
        addInstance(with: glassCubeGeometry,
                    translation: SIMD3<Float>(2.5, 0.5, -2.25),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(1.0, 1.0, 1.0),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        // Torus around glass cube
        addInstance(with: torusGeometry,
                    translation: SIMD3<Float>(2.5, 0.5, -2.25),
                    rotation: SIMD3<Float>(.pi/2, 0, 0),
                    scale: SIMD3<Float>(0.6, 0.2, 0.6),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // Glass ball (center)
        addInstance(with: glassBallGeometry,
                    translation: SIMD3<Float>(2.5, 0.75, -0.25),
                    scale: SIMD3<Float>(1.5, 1.5, 1.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        // Plastic ball inside glass ball
        addInstance(with: plasticBallGeometry,
                    translation: SIMD3<Float>(2.5, 0.75, -0.25),
                    scale: SIMD3<Float>(0.55, 0.55, 0.55),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // Side mirror
        addInstance(with: mirrorCubeGeometry,
                    translation: SIMD3<Float>(0.9, 2, 0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.001, 1, 1),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        // Glass ball (front right)
        addInstance(with: glassBallGeometry,
                    translation: SIMD3<Float>(2.5, 0.75, 2.0),
                    scale: SIMD3<Float>(1.5, 1.5, 1.5),
                    mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        // Light balls
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(0.0, 2.5, -3),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(2.0, 2.5, -3),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5))
    }

    func createMaterialScene() {
        cameraLocations = [(SIMD3<Float>(0.0, 3.95, 0.0), SIMD3<Float>(0.0, 1.0, -0.01)),
                           (SIMD3<Float>(2.9, 1.5, 2.9), SIMD3<Float>(0.41862917, 0.19517994, -0.13483834)),
                           (SIMD3<Float>(2.5309057, 0.53865755, 2.1793683), SIMD3<Float>(1.1135497, -1.1510044, -1.314362)),
                           (SIMD3<Float>(-2.362045, 1.0173429, -0.12236312), SIMD3<Float>(-0.58842844, 0.92927957, 3.6081252)),
                           (SIMD3<Float>(-0.50906193, 1.4787455, 1.1016287), SIMD3<Float>(0.41320705, 0.41911006, 3.6956992)),
                           (SIMD3<Float>(1.5, 1.6999998, 0.0), SIMD3<Float>(1.5, -1.25, -0.01)),
                           (SIMD3<Float>(2.1914167, 2.0363708, 1.2903469), SIMD3<Float>(0.7327032, 0.97673523, 3.6252854)),
                           (SIMD3<Float>(0.63316226, 0.11427891, -0.70196056), SIMD3<Float>(2.8992136, 0.6142791, 2.7163746)),
                           (SIMD3<Float>(2.6188042, 0.4169299, 2.4784868), SIMD3<Float>(1.2169056, 0.8069375, -0.08767253)),
                           (SIMD3<Float>(3.693739, 21.429874, -13.260839), SIMD3<Float>(3.305217, 20.370247, -10.535254))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[2]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildBox(width: 6.0, height: 4.0, depth: 6.0)
        
        let dragonGeometry = addAssimpGeometry(fileName: "stanford_dragon", fileExtension: "obj", defaultMaterial: GLASS)
        let angelGeometry = addAssimpGeometry(fileName: "lucy", fileExtension: "obj", defaultMaterial: GLASS)
        let saintGeometry = addAssimpGeometry(fileName: "saint", fileExtension: "obj", defaultMaterial: GLASS)
                
        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(-1.8, 0.0, 2.5),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(0.1, 0.1, 0.1)
        )
        
        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(-1.8, 2.0, 2.5),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(1, 1, 1)
        )
        
        addInstance(with: angelGeometry,
                    translation: SIMD3<Float>(0, 0, 2.5),
                    rotation: SIMD3<Float>(.pi, 0, 0),
                    scale: SIMD3<Float>(0.0015, 0.0015, 0.0015)
        )

        addInstance(with: saintGeometry,
                            translation: SIMD3<Float>(1.5, 0, 2.2),
                            rotation: SIMD3<Float>(-.pi/2, 0, 0),
                            scale: SIMD3<Float>(0.01, 0.01, 0.01)
                )

        var plasticMaterial = PLASTIC
        
        for i in 0...10 {
            plasticMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingPlasticGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: plasticMaterial)
            
            addInstance(with: scalingPlasticGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, -1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
        
        var mirrorMaterial = MIRROR
        
        for i in 0...10 {
            mirrorMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingMirrorGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: mirrorMaterial)
            
            addInstance(with: scalingMirrorGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 0.0),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
                
        var glassMaterial = GLASS
        
        for i in 0...10 {
            glassMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingGlassGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: glassMaterial)
            
            addInstance(with: scalingGlassGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }

        var coloredGlassMaterial = GLASS

        for i in 0...10 {
            coloredGlassMaterial.colorValue = colors[i]

            let coloredGlassGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: coloredGlassMaterial)

            addInstance(with: coloredGlassGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 2.0, 2.8),
                        scale: SIMD3<Float>(0.4, 0.1, 0.4)
            )
        }
        
        let lightStripGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj",
                                                   defaultMaterial: createEmissiveMaterial(color: .one), emissionAmplifier: 10.0)

        addInstance(with: lightStripGeometry,
                    translation: SIMD3<Float>(0.0, 3.95, -1.0),
                    scale: SIMD3<Float>(3.99, 0.01, 0.8))
    
        addInstance(with: lightStripGeometry,
                    translation: SIMD3<Float>(0.0, 3.95, 1.0),
                    scale: SIMD3<Float>(3.99, 0.01, 0.8))
    }
    
    func createMISScene() {
        cameraLocations = [
            (SIMD3<Float>(0.0, 2.25, 2.95), SIMD3<Float>(0.0, 1.4, 0.0))
        ]

        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        buildBox(width: 6.0, height: 4.0, depth: 6.0)

        var mirrorMaterial = MIRROR
        let roughnesses: [Float] = [0.5, 0.3, 0.1, 0.0]
        let elevations: [Float] = [0.5, 0.7, 1.05, 1.55]
        let rotations: [Float] = [.pi/24, .pi/12, .pi/6, .pi/3.1]

        for i in 0..<4 {
            mirrorMaterial.roughnessValue = roughnesses[i]

            let mirrorSlabGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj", defaultMaterial: mirrorMaterial)

            addInstance(with: mirrorSlabGeometry,
                        translation: SIMD3<Float>(0, elevations[i], 0.5 - 0.85 * Float(i)),
                        rotation: SIMD3<Float>(rotations[i], 0, 0),
                        scale: SIMD3<Float>(4.0, 0.05, 0.75))
        }
    
        let sizes: [Float] = [0.025, 0.15, 0.4, 1.0]

        for i in 0..<4 {
            let lightBallGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj",
                                                      defaultMaterial: createEmissiveMaterial(color: colors[i * 3]),
                                                      emissionAmplifier: 200.0)

            addInstance(with: lightBallGeometry,
                        translation: SIMD3<Float>(-1.5 + Float(i), 2.5, -1.5),
                        scale: sizes[i] * SIMD3<Float>(1.0, 1.0, 1.0))
        }
    }
        
    func createBistroScene() {
        cameraLocations = [
            (SIMD3<Float>(-14.601961, 3.1426682, -1.5453186), SIMD3<Float>(-5.979579, 1.3546469, 0.31400716)),
            (SIMD3<Float>(47.638798, 3.7606301, 37.219215), SIMD3<Float>(40.495815, 3.7606497, 31.744099))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let bistroGeometry = addGLTFGeometry(fileName: "Bistro_Godot", fileExtension: "glb", emissionAmplifier: 0)
        addInstance(with: bistroGeometry)
        
        addEnvironmentMap(textureURL: skyURL)
    }
    
    func createCornellScene() {
        cameraLocations = [
            (SIMD3<Float>(-1.0515742e-08, 2.1341612, -2.5105686), SIMD3<Float>(-6.5451777e-09, 1.8385572, -1.5556743))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let cornellBoxGeometry = addGLTFGeometry(fileName: "veach_mis_remake", fileExtension: "glb", emissionAmplifier: 10)
        addInstance(with: cornellBoxGeometry
//                    rotation: SIMD3<Float>(-.pi/2, 0, .pi/2)
        )
    }
    
    func createTheWhiteRoom() {
        cameraLocations = [
            (SIMD3<Float>(2.617519, 1.4286628, 6.338271), SIMD3<Float>(2.0222836, 1.3286594, 5.541462))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let theWhiteRoomGeometry = addGLTFGeometry(fileName: "the-white-room", fileExtension: "gltf", emissionAmplifier: 5)
        addInstance(with: theWhiteRoomGeometry)
    }


    
    func createKoenigseggScene() {
        cameraLocations = [(SIMD3<Float>(0, 3, 9), SIMD3<Float>(0, 3, 0)),
                           (SIMD3<Float>(2.8985946, 1.0271178, -1.9204926), SIMD3<Float>(-3.6789668, -0.76090455, 3.9564571)),
                           (SIMD3<Float>(0.17043182, 0.72948277, -0.16555624), SIMD3<Float>(0.6853374, -1.058545, 8.639974))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let whiteCubeGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj")
        addInstance(with: whiteCubeGeometry, scale: SIMD3<Float>(6, 0.1, 6))

        let koenigseggGeometry = addGLTFGeometry(fileName: "koenigsegg_one_pro", fileExtension: "glb", emissionAmplifier: 5.0)
        addInstance(with: koenigseggGeometry, scale: 10 * .one)

        addEnvironmentMap(textureURL: duskURL)
    }
    
    func createSoloScene() {
        cameraLocations = [(SIMD3<Float>(0, 3, 9), SIMD3<Float>(0, 3, 0)),
                           (SIMD3<Float>(2.8985946, 1.0271178, -1.9204926), SIMD3<Float>(-3.6789668, -0.76090455, 3.9564571)),
                           (SIMD3<Float>(0.17043182, 0.72948277, -0.16555624), SIMD3<Float>(0.6853374, -1.058545, 8.639974))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
//        let dragonGeometry = addAssimpGeometry(fileName: "stanford_dragon", fileExtension: "obj", defaultMaterial: GLASS)
//        let angelGeometry = addAssimpGeometry(fileName: "lucy", fileExtension: "obj", defaultMaterial: GLASS)
//        let saintGeometry = addAssimpGeometry(fileName: "saint", fileExtension: "obj", defaultMaterial: GLASS)
//                
//        addInstance(with: dragonGeometry,
//                    translation: SIMD3<Float>(-1.8, 0.0, 2.5),
//                    rotation: SIMD3<Float>(0, .pi, 0),
//                    scale: SIMD3<Float>(0.1, 0.1, 0.1)
//        )
//        
//        addInstance(with: dragonGeometry,
//                    translation: SIMD3<Float>(-1.8, 2.0, 2.5),
//                    rotation: SIMD3<Float>(0, .pi, 0),
//                    scale: SIMD3<Float>(1, 1, 1)
//        )
//        
//        addInstance(with: angelGeometry,
//                    translation: SIMD3<Float>(0, 0, 2.5),
//                    rotation: SIMD3<Float>(.pi, 0, 0),
//                    scale: SIMD3<Float>(0.0015, 0.0015, 0.0015)
//        )
//
//        addInstance(with: saintGeometry,
//                    translation: SIMD3<Float>(1.5, 0, 2.2),
//                    rotation: SIMD3<Float>(-.pi/2, 0, 0),
//                    scale: SIMD3<Float>(0.01, 0.01, 0.01)
//                )

        let scalingPlasticGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: SMOOTH_OPAQUE_DIELECTRIC)
        
        addInstance(with: scalingPlasticGeometry,
                    translation: SIMD3<Float>(-20, 0, 0),
                    scale: 25 * .one
        )
        
        addInstance(with: scalingPlasticGeometry,
                    translation: SIMD3<Float>(20, 0, 0),
                    scale: 25 * .one
        )

        addEnvironmentMap(textureURL: skyURL)
    }

         
    func createEnvironmentMapBallsScene() {
        cameraLocations = [(SIMD3<Float>(0.0, 3.95, 0.0), SIMD3<Float>(0.0, 1.0, -0.01)),
                           (SIMD3<Float>(-4.7358356, 0.8316741, -0.24047767), SIMD3<Float>(-1.8881888, 1.2216821, 0.42390215)),
                           (SIMD3<Float>(3.3002243, 1.739783, 3.0232406), SIMD3<Float>(1.4852387, -0.08613372, 1.5829433)),
                           (SIMD3<Float>(-0.58075345, 1.6034509, 3.8094723), SIMD3<Float>(-0.05752662, 0.27425483, 1.2283735)),
                           (SIMD3<Float>(-0.25849676, 1.2781403, -1.3962265), SIMD3<Float>(0.80470467, -0.3073362, 0.8528768)),
                           (SIMD3<Float>(0.9514366, 1.4333981, 3.8511183), SIMD3<Float>(0.42823184, 0.10420242, 1.2700162)),
                           (SIMD3<Float>(-2.2035198, 1.7945361, -4.0881658), SIMD3<Float>(-1.0779829, 0.4653412, -1.7072012))
        ]
        
        (cameraPosition, cameraTarget) = cameraLocations[6]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let whiteCubeGeometry = addAssimpGeometry(fileName: "cube", fileExtension: "obj")
        addInstance(with: whiteCubeGeometry, scale: SIMD3<Float>(6, 0.1, 6))
        
        var plasticMaterial = PLASTIC
        
        for i in 0...10 {
            plasticMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingPlasticGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: plasticMaterial)
            
            addInstance(with: scalingPlasticGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, -1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
        
        var mirrorMaterial = MIRROR
        
        for i in 0...10 {
            mirrorMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingMirrorGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: mirrorMaterial)
            
            addInstance(with: scalingMirrorGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 0.0),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
                
        var glassMaterial = GLASS
        
        for i in 0...10 {
            glassMaterial.roughnessValue = Float(i) * 0.1
            
            let scalingGlassGeometry = addAssimpGeometry(fileName: "ball", fileExtension: "obj", defaultMaterial: glassMaterial)
            
            addInstance(with: scalingGlassGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
        
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(0.1, 0.0, 0.0))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(0, 10, 0),
                    scale: 0.01 * SIMD3<Float>(1, 1, 1)
        )
        
        addEnvironmentMap(textureURL: skyURL)
//        addDirectionalLight(direction: simd_normalize(SIMD3<Float>(0.0, -0.5, 1.0)), color: SIMD3<Float>(10, 10, 10))
//        addDirectionalLight(direction: simd_normalize(SIMD3<Float>(1.0, -0.5, 1.0)), color: SIMD3<Float>(0, 10, 0))
//        addDirectionalLight(direction: simd_normalize(SIMD3<Float>(-1.0, -0.5, 1.0)), color: SIMD3<Float>(0, 0, 10))
//        ddDirectionalLight(direction: simd_normalize(SIMD3<Float>(0.5, -0.5, 0.75)), color: 10 * .one)
    }

    func addInstance(with geometry: Geometry, translation: SIMD3<Float> = .zero, rotation: SIMD3<Float> = .zero, scale: SIMD3<Float> = .one,
                     mask: UInt32 = GEOMETRY_MASK_OPAQUE) {
                        
        let opaqueInstance = GeometryInstance(geometry: geometry,
                                              translation: translation,
                                              rotation: rotation,
                                              scale: scale,
                                              mask: GEOMETRY_MASK_TRIANGLE | mask)
        addInstance(opaqueInstance)
        
        instanceLightIndices.append(Int32(lights.count))
        
        for light in geometry.areaLights {
            let (instanceLightTriangles, totalArea) = getLightTriangles(areaLight: light, transform: opaqueInstance.transform)

            let areaLight = Light(type: AREA_LIGHT,
                                  index: UInt32(instances.count - 1),
                                  delta: false,
                                  position: .zero,
                                  color: light.averageEmission,
                                  firstTriangleIndex: UInt32(lightTriangles.count),
                                  triangleCount: UInt32(instanceLightTriangles.count),
                                  totalArea: totalArea,
                                  direction: .zero
            )

            lightTriangles.append(contentsOf: instanceLightTriangles)
            lights.append(areaLight)
            print("new light", light.averageEmission, totalArea, length(light.averageEmission / totalArea), instanceLightTriangles.count, lightTriangles[0].emission)
        }
    }
    
    func getLightTriangles(areaLight: AreaLight, transform: simd_float4x4) -> ([LightTriangle], Float) {
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
                                                        emissionTextureIndex: areaLight.emissionTextureIndex,
                                                        CDF: totalArea)
            )
        }
        
        for i in 0..<instanceLightTriangles.count {
            instanceLightTriangles[i].CDF /= totalArea
        }
        
        return (instanceLightTriangles, totalArea)
    }
            
    func addPointLight(position: SIMD3<Float>, color: SIMD3<Float>) {
        let newPointLight = Light(type: POINT_LIGHT,
                                  index: UInt32(lights.count),
                                  delta: true,
                                  position: position,
                                  color: color,
                                  firstTriangleIndex: 0,
                                  triangleCount: 0,
                                  totalArea: 0.0,
                                  direction: .zero
        )
        
        lights.append(newPointLight)
    }
    
    func addDirectionalLight(direction: SIMD3<Float>, color: SIMD3<Float>) {
        let newDirectionalLight = Light(type: DIRECTIONAL_LIGHT,
                                        index: UInt32(lights.count),
                                        delta: true,
                                        position: .zero,
                                        color: color,
                                        firstTriangleIndex: 0,
                                        triangleCount: 0,
                                        totalArea: 0,
                                        direction: direction)
        
        lights.append(newDirectionalLight)
    }
    
    func addEnvironmentMap(textureURL: URL) {
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [.SRGB: true, .textureStorageMode: MTLStorageMode.shared.rawValue]
        
        do {
            environmentMapTexture = try textureLoader.newTexture(URL: textureURL, options: options)
        } catch {
            fatalError("Couldn't load environemntMap texture texture: \(error)")
        }
        print(environmentMapTexture?.width, environmentMapTexture?.height, environmentMapTexture?.pixelFormat)
        
        let (CDF, averageColor) = getEnvironmentMapCDF(texture: environmentMapTexture!)
        environmentMapCDF = CDF
        
        let newEnvironmentLight = Light(type: ENVIRONMENT_MAP,
                                        index: UInt32(lights.count),
                                        delta: false,
                                        position: .zero,
                                        color: averageColor,
                                        firstTriangleIndex: 0,
                                        triangleCount: 0,
                                        totalArea: 0,
                                        direction: .zero)

        lights.append(newEnvironmentLight)
    }
    
    func getEnvironmentMapCDF(texture: MTLTexture) -> ([Float], SIMD3<Float>) {
        let width = texture.width
        let height = texture.height
        var pixels = [Float](repeating: 0, count: width * height * 4)
        var averageRGB: SIMD3<Float> = .zero
        
        texture.getBytes(&pixels,
                         bytesPerRow: width * MemoryLayout<Float>.size * 4,
                         from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1)),
                         mipmapLevel: 0)
        
        var CDF = [Float](repeating: 0, count: width * height)
        var totalLuminance: Float = 0.0
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * width + x) * 4
                let r = pixels[idx]
                let g = pixels[idx + 1]
                let b = pixels[idx + 2]
                
                averageRGB.x += r
                averageRGB.y += g
                averageRGB.z += b
                
                let v = (Float(y) + 0.5) / Float(height)
                let theta = v * .pi
                let sinTheta = sin(theta)

                let luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) * sinTheta
                CDF[y * width + x] = luminance + (y * width + x > 0 ? CDF[y * width + x - 1] : 0.0)
                totalLuminance += luminance
            }
        }
        
//        print("Total Luminance:", totalLuminance)
        averageRGB /= Float(width * height)
        
        for i in 0..<height * width {
            CDF[i] /= totalLuminance
        }
        
        return (CDF, averageRGB)
    }
        
    func addAssimpGeometry(fileName: String,
                           fileExtension: String,
                           defaultMaterial: Material? = nil,
                           defaultTexture: TextureInfo = TextureInfo(),
                           emissionAmplifier: Float = 1.0
                           ) -> AssimpGeometry
    {
        if let modelPath = Bundle.main.path(forResource: fileName, ofType: fileExtension) {
            
            let geometry = AssimpGeometry(device: device,
                                          modelPath: modelPath,
                                          defaultMaterial: defaultMaterial,
                                          defaultTexture: defaultTexture,
                                          emissionAmplifier: emissionAmplifier)
            addGeometry(geometry)
            return geometry
        } else {
            fatalError("Failed to find resource \(fileName)")
        }
    }
    
    func addGLTFGeometry(
        fileName: String,
        fileExtension: String,
        emissionAmplifier: Float = 1.0
    ) -> GLTFGeometry {
        guard let modelPath = Bundle.main.path(forResource: fileName, ofType: fileExtension) else {
            fatalError("[GameScene] Failed to find resource \(fileName).\(fileExtension)")
        }
 
        let geometry = GLTFGeometry(
            device: device,
            modelPath: modelPath,
            emissionAmplifier: emissionAmplifier
        )
        addGeometry(geometry)
        return geometry
    }
}
