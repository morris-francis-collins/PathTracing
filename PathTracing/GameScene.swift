//
//  GameScene.swift
//  PathTracing
//
//  Created on 3/22/25.
//

import MetalKit
import simd

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
    var lights: [AreaLight] = []
    var lightBuffer: MTLBuffer?
    var lightTriangles: [LightTriangle] = []
    var lightTriangleBuffer: MTLBuffer?
    var textures: [MTLTexture] = []
    
    var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
    var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var cameraUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    
    var cameraSpeed: Float = 0.25
    var rotationSpeed: Float = 0.05
    
    var cameraLocations: [(SIMD3<Float>, SIMD3<Float>)] = []
    
    init(device: MTLDevice) {
        self.device = device
        createDifficultScene()
    }
    
    func clear() {
        geometries.removeAll()
        instances.removeAll()
        lights.removeAll()
    }
    
    func addGeometry(_ mesh: Geometry) {
        geometries.append(mesh)
    }
    
    func addInstance(_ instance: GeometryInstance) {
        instances.append(instance)
    }
    
    func addLight(_ light: AreaLight) {
        lights.append(light)
    }
        
    func uploadToBuffers() {
        for geometry in geometries {
            geometry.uploadToBuffers()
        }
        
        let options = getManagedBufferStorageMode()

        lightBuffer = device.makeBuffer(bytes: lights, length: lights.count * MemoryLayout<AreaLight>.size, options: options)
        lightTriangleBuffer = device.makeBuffer(bytes: lightTriangles, length: lightTriangles.count * MemoryLayout<LightTriangle>.size, options: options)
        
        #if !os(iOS)
        lightBuffer?.didModifyRange(0..<lightBuffer!.length)
        lightTriangleBuffer?.didModifyRange(0..<lightTriangleBuffer!.length)
        #endif
    }
    
    func buildSimpleBox() { // x coordinates are flipped; left is positive, right is negative
        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL, textureURL: checkerBoardURL)
        wallGeometry.uploadToBuffers()
        addGeometry(wallGeometry)
        
        let whiteCubeGeometry = ObjGeometry(device: device, objURL: cubeURL)
        whiteCubeGeometry.uploadToBuffers()
        addGeometry(whiteCubeGeometry)
                
        let redCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 0, 0))
        redCubeGeometry.uploadToBuffers()
        addGeometry(redCubeGeometry)

        let greenCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(0, 1, 0))
        greenCubeGeometry.uploadToBuffers()
        addGeometry(greenCubeGeometry)
        
        let floorInstance = GeometryInstance(
            geometry: whiteCubeGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(4.0, 0.4, 4.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let backWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0.0, 1.8, 2.2),
            rotation: SIMD3<Float>(0, 0, .pi/2),
            scale: SIMD3<Float>(4.0, 4.0, 0.4),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let leftWallInstance = GeometryInstance(
            geometry: redCubeGeometry,
            translation: SIMD3<Float>(2.2, 1.8, 0.0),
            rotation: SIMD3<Float>(0, .pi/2, .pi/2),
            scale: SIMD3<Float>(4.0, 4.0, 0.4),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let rightWallInstance = GeometryInstance(
            geometry: greenCubeGeometry,
            translation: SIMD3<Float>(-2.2, 1.8, 0.0),
            rotation: SIMD3<Float>(0, .pi/2, .pi/2),
            scale: SIMD3<Float>(4.0, 4.0, 0.4),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let ceilingInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0, 3.6, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(4.0, 0.4, 4.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        addInstance(floorInstance)
        addInstance(backWallInstance)
        addInstance(leftWallInstance)
        addInstance(rightWallInstance)
        addInstance(ceilingInstance)
    }
    
    func buildSegmentedBox(extra: Bool = false) {
        let width: Float = 8.0
        let height: Float = 4.0
        let depth: Float = 8.0
        let epsilon: Float = 1e-3
        
        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL)
        wallGeometry.uploadToBuffers()
        addGeometry(wallGeometry)
        
        let mirrorCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        mirrorCubeGeometry.uploadToBuffers()
        addGeometry(mirrorCubeGeometry)
        
        let floorInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let leftWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(-width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let rightWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let backWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
                
        let ceilingInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0.0, height, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let middleWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(1.0, height / 2, -1.5),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(100 * epsilon, height, 5 * depth / 8),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let rightMirrorInstance = GeometryInstance(
            geometry: mirrorCubeGeometry,
            translation: SIMD3<Float>(2.5, 2.0, -3.9),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(2.0, 4.0, 0.01),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let leftMirrorInstance = GeometryInstance(
            geometry: mirrorCubeGeometry,
            translation: SIMD3<Float>(-3.0, 2.0, -3.9),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(1.0, 3.0, 0.01),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        addInstance(floorInstance)
        addInstance(leftWallInstance)
        addInstance(rightWallInstance)
        addInstance(backWallInstance)
        addInstance(ceilingInstance)
        addInstance(middleWallInstance)
        addInstance(rightMirrorInstance)
        addInstance(leftMirrorInstance)
        
        if (!extra) { return }
        
        let leftSideMirrorInstance = GeometryInstance(geometry: mirrorCubeGeometry,
                                                  translation: SIMD3<Float>(0.9, 2, 0),
                                                  scale: SIMD3<Float>(0.001, 1, 1),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let rightSideMirrorInstance1 = GeometryInstance(geometry: mirrorCubeGeometry,
                                                        translation: SIMD3<Float>(1.1, 2, -2.25),
                                                        scale: SIMD3<Float>(0.001, 4, 1.5),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        let rightSideMirrorInstance2 = GeometryInstance(geometry: mirrorCubeGeometry,
                                                        translation: SIMD3<Float>(1.1, 2, -0.25),
                                                        scale: SIMD3<Float>(0.001, 4, 1.5),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        addInstance(leftSideMirrorInstance)
        addInstance(rightSideMirrorInstance1)
        addInstance(rightSideMirrorInstance2)
    }
        
    func createScene() { // x coordinates are flipped; left is positive, right is negative
        cameraPosition = SIMD3<Float>(0.0, 2.0, -6.0)
        cameraTarget = SIMD3<Float>(0.0, 1.75, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        buildSimpleBox()


        let whiteCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
        whiteCubeGeometry.uploadToBuffers()
        addGeometry(whiteCubeGeometry)
        
        let ballGeometry = ObjGeometry(device: device, objURL: ballURL, material: GLASS)
        ballGeometry.uploadToBuffers()
        addGeometry(ballGeometry)
        
        let pyramidGeometry = ObjGeometry(device: device, objURL: pyramidURL, material: GLASS)
        pyramidGeometry.uploadToBuffers()
        addGeometry(pyramidGeometry)
                
        let tableGeometry = ModelIOGeometry(device: device, modelURL: tableURL)
        tableGeometry.uploadToBuffers()
        addGeometry(tableGeometry)

        let cubeInstance = GeometryInstance(
            geometry: whiteCubeGeometry,
            translation: SIMD3<Float>(-0.0, 0.2 + 0.4 + 0.7, -0.1),
            rotation: SIMD3<Float>(0, -.pi/7, 0),
            scale: SIMD3<Float>(0.8, 0.8, 0.8),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )

        let ballInstance = GeometryInstance(
            geometry: ballGeometry,
            translation: SIMD3<Float>(0, 0.9, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(1, 1, 1),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )
        
        let rightBallInstance = GeometryInstance(
            geometry: ballGeometry,
            translation: SIMD3<Float>(-1.2, 1.2, -0.5),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(1, 1, 1),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )
        
        let leftBallInstance = GeometryInstance(
            geometry: ballGeometry,
            translation: SIMD3<Float>(1.2, 0.8, 0.8),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(1, 1, 1),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )

        
        let pyramidInstance = GeometryInstance(
            geometry: pyramidGeometry,
            translation: SIMD3<Float>(0.9, 0.3, 0.0),
            rotation: SIMD3<Float>(0, -.pi/6, 0),
            scale: SIMD3<Float>(0.8, 0.8, 0.8),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )
        
        let tableInstance = GeometryInstance(
            geometry: tableGeometry,
            translation: SIMD3<Float>(0.0, 1.5, 0.0),
            rotation: SIMD3<Float>(0, .pi/4, 0),
            scale: SIMD3<Float>(2.0, 1.0, 1.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

//        addInstance(cubeInstance)
        addInstance(ballInstance)
//        addInstance(rightBallInstance)
//        addInstance(leftBallInstance)
//        addInstance(pyramidInstance)
        addInstance(tableInstance)

        let leftLightGeometry = ObjGeometry(device: device, objURL: cubeURL, color: .zero)
        leftLightGeometry.uploadToBuffers()
        addGeometry(leftLightGeometry)
        
        let leftLightInstance = GeometryInstance(geometry: whiteCubeGeometry,
                                                 translation: SIMD3<Float>(0.0, 3.39, -1.0),
                                                 scale: SIMD3<Float>(3.9, 0.00001, 0.8),
                                                 mask: GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_OPAQUE)
        
        addLight(lightInstance: leftLightInstance, lightForward: SIMD3<Float>(0.0, -1.0, 0.0))
        
        let rightLightGeometry = ObjGeometry(device: device, objURL: cubeURL, color: .zero)
        rightLightGeometry.uploadToBuffers()
        addGeometry(rightLightGeometry)
        
        let rightLightInstance = GeometryInstance(geometry: whiteCubeGeometry,
                                                 translation: SIMD3<Float>(0.0, 3.39, 1.0),
                                                  scale: SIMD3<Float>(3.9, 0.00001, 0.8),
                                                 mask: GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_OPAQUE)
        
        addLight(lightInstance: rightLightInstance, lightForward: SIMD3<Float>(0.0, -1.0, 0.0))
    }
    
    func createLivelyScene() {
//        cameraPosition = SIMD3<Float>(0, 2.5, 10)
//        cameraTarget = SIMD3<Float>(0.0, 2, 0.0)
        cameraPosition = SIMD3<Float>(1, 2.5, 6)
        cameraTarget = SIMD3<Float>(0.0, 2, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildSegmentedBox()

        let cubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
        cubeGeometry.uploadToBuffers()
        addGeometry(cubeGeometry)
        
        let tableGeometry = ObjGeometry(device: device, objURL: tableURL, color: SIMD3<Float>(1, 1, 1), material: PLASTIC)
        tableGeometry.uploadToBuffers()
        addGeometry(tableGeometry)
        
        let plasticBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: PLASTIC)
        plasticBallGeometry.uploadToBuffers()
        addGeometry(plasticBallGeometry)
        
        let glassBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
        glassBallGeometry.uploadToBuffers()
        addGeometry(glassBallGeometry)
        
        let mirrorBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        mirrorBallGeometry.uploadToBuffers()
        addGeometry(mirrorBallGeometry)
        
        let mirrorGeometry = ModelIOGeometry(device: device, modelURL: mirrorURL)
        mirrorGeometry.uploadToBuffers()
        addGeometry(mirrorGeometry)
        
        let teaTableGeometry = ModelIOGeometry(device: device, modelURL: teaTableURL)
        teaTableGeometry.uploadToBuffers()
        addGeometry(teaTableGeometry)
        
        let couchGeometry = ModelIOGeometry(device: device, modelURL: couchURL)
        couchGeometry.uploadToBuffers()
        addGeometry(couchGeometry)
        
        let hangingLightGeometry = ModelIOGeometry(device: device, modelURL: hangingLightURL)
        
        let wallLightGeometry = ModelIOGeometry(device: device, modelURL: wallLightURL)
        
        let floorLampGeometry = ModelIOGeometry(device: device, modelURL: floorLampURL)
            
        let tableInstance = GeometryInstance(
            geometry: tableGeometry,
            translation: SIMD3<Float>(2.5, 0.0, -1.5),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(2.0, 1.5, 4.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let glassBallInstance = GeometryInstance(
            geometry: glassBallGeometry,
            translation: SIMD3<Float>(2.5, 1.5 + 0.375, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(0.75, 0.75, 0.75),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let glassrightBallInstance = GeometryInstance(
            geometry: glassBallGeometry,
            translation: SIMD3<Float>(0.0, 0.5, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(0.5, 0.5, 0.5),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let mirrorBallInstance = GeometryInstance(
            geometry: mirrorBallGeometry,
            translation: SIMD3<Float>(-1.5, 1.1, -0.75),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(0.5, 0.5, 0.5),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let mirrorInstance = GeometryInstance(
            geometry: mirrorGeometry,
            translation: SIMD3<Float>(-0.5, 1.75, -2),
            rotation: SIMD3<Float>(0, -.pi/8, 0),
            scale: SIMD3<Float>(0.02, 0.012, 0.01),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let teaTableInstance = GeometryInstance(
            geometry: teaTableGeometry,
            translation: SIMD3<Float>(-1.5, 0, 0.0),
            rotation: SIMD3<Float>(0, .pi, 0),
            scale: SIMD3<Float>(2, 2, 2),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let couchInstance = GeometryInstance(
            geometry: couchGeometry,
            translation: SIMD3<Float>(-3.15, 0, 0),
            rotation: SIMD3<Float>(0, .pi/2, 0),
            scale: SIMD3<Float>(0.017, 0.017, 0.017),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        addInstance(with: hangingLightGeometry,
                    translation: SIMD3<Float>(-0.5, -0.25, 2.0),
                    rotation: SIMD3<Float>(0, .pi/2, 0),
                    scale: SIMD3<Float>(0.005, 0.005, 0.005),
                    lightAmplifier: 5.0,
                    mask: GEOMETRY_MASK_OPAQUE)
        
        addInstance(with: wallLightGeometry,
                    translation: SIMD3<Float>(1.25, 2.5, 0.0),
                    rotation: SIMD3<Float>(0, .pi/2 + 0.55, 0),
                    scale: SIMD3<Float>(0.005, 0.005, 0.005),
                    lightAmplifier: 3.0,
                    mask: GEOMETRY_MASK_OPAQUE)
        
        addInstance(with: floorLampGeometry,
                    translation: SIMD3<Float>(-3.5, 0.0, -3.5),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(0.015, 0.015, 0.015),
                    lightAmplifier: 3.0,
                    mask: GEOMETRY_MASK_OPAQUE)

        addInstance(tableInstance)
        addInstance(glassBallInstance)
        addInstance(glassrightBallInstance)
        addInstance(mirrorBallInstance)
        addInstance(mirrorInstance)
        addInstance(teaTableInstance)
        addInstance(couchInstance)
    }
            
    func createDifficultScene() {
//        cameraPosition = SIMD3<Float>(-1.9, 3.5, 0.5)
//        cameraTarget = SIMD3<Float>(-2.0, 1, 0.0)
        cameraPosition = SIMD3<Float>(0, 2.5, 6)
        cameraTarget = SIMD3<Float>(0, 2, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        cameraLocations = [(cameraPosition, cameraTarget),
                            (SIMD3<Float>(2.9280746, 2.5814812, 3.0649862), SIMD3<Float>(-0.489712, 1.8794974, -1.8417473)),
                            (SIMD3<Float>(3.860585, 2.7261002, -0.094045304), SIMD3<Float>(-0.15811086, -0.75679064, -2.9171062)),
                            (SIMD3<Float>(3.860585, 1.2261002, 0.9059547), SIMD3<Float>(-0.120718956, -1.2116656, -2.896234))
                          ]
        
        buildSegmentedBox(extra: true)
        
        let plasticBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: PLASTIC)
        plasticBallGeometry.uploadToBuffers()
        addGeometry(plasticBallGeometry)
        
        let glassBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
        glassBallGeometry.uploadToBuffers()
        addGeometry(glassBallGeometry)
        
        let mirrorBallGeometry = ObjGeometry(device: device, objURL: ballURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        mirrorBallGeometry.uploadToBuffers()
        addGeometry(mirrorBallGeometry)
        
        let mirrorCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        mirrorCubeGeometry.uploadToBuffers()
        addGeometry(mirrorCubeGeometry)
                
        let ringGeometry = ObjGeometry(device: device, objURL: ringURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        ringGeometry.uploadToBuffers()
        addGeometry(ringGeometry)
        
        let glassCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
        glassCubeGeometry.uploadToBuffers()
        addGeometry(glassCubeGeometry)
        
        let torusGeometry = ObjGeometry(device: device, objURL: torusURL, color: SIMD3<Float>(1, 1, 1), material: PLASTIC)
        torusGeometry.uploadToBuffers()
        addGeometry(torusGeometry)


        
        let ringInstance = GeometryInstance(geometry: ringGeometry,
                                            translation: SIMD3<Float>(-2, 0.25, 0),
                                            scale: SIMD3<Float>(2.0, 0.5, 2.0),
                                            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let ringInstance2 = GeometryInstance(geometry: ringGeometry,
                                             translation: SIMD3<Float>(-3.75, 2, 0),
                                             rotation: SIMD3<Float>(0, 0, .pi/2),
                                             scale: SIMD3<Float>(2.0, 0.5, 2.0),
                                             mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let glassCubeInstance = GeometryInstance(geometry: glassCubeGeometry,
                                                 translation: SIMD3<Float>(2.5, 0.5, -2.25),
                                                 rotation: SIMD3<Float>(0, 0, 0),
                                                 scale: SIMD3<Float>(1.0, 1.0, 1.0),
                                                 mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        let torusInstance = GeometryInstance(geometry: torusGeometry,
                                             translation: SIMD3<Float>(2.5, 0.5, -2.25),
                                             rotation: SIMD3<Float>(.pi/2, 0, 0),
                                             scale: SIMD3<Float>(0.6, 0.2, 0.6),
                                             mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        let glassBallInstance = GeometryInstance(geometry: glassBallGeometry,
                                                 translation: SIMD3<Float>(2.5, 0.5, -0.25),
                                                 scale: SIMD3<Float>(1.5, 1.5, 1.5),
                                                 mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        let plasticBallInstance = GeometryInstance(geometry: plasticBallGeometry,
                                                   translation: SIMD3<Float>(2.5, 0.5, -0.25),
                                                   scale: SIMD3<Float>(0.75, 0.75, 0.75),
                                                   mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let sideMirrorInstance = GeometryInstance(geometry: mirrorCubeGeometry,
                                                  translation: SIMD3<Float>(0.9, 2, 0),
                                                  rotation: SIMD3<Float>(0, 0, 0),
                                                  scale: SIMD3<Float>(0.001, 1, 1),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let frontRightGlassBallInstance = GeometryInstance(geometry: glassBallGeometry,
                                                           translation: SIMD3<Float>(2.5, 0.5, 2.0),
                                                            scale: SIMD3<Float>(1.5, 1.5, 1.5),
                                                            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)

        
        addInstance(ringInstance)
        addInstance(ringInstance2)
        addInstance(glassCubeInstance)
        addInstance(torusInstance)
        addInstance(glassBallInstance)
        addInstance(plasticBallInstance)
        addInstance(sideMirrorInstance)
        addInstance(frontRightGlassBallInstance)
        
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 1.0))

        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(0.0, 2.5, -3), scale: SIMD3<Float>(0.5, 0.5, 0.5))
        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(2.0, 2.5, -3), scale: SIMD3<Float>(0.5, 0.5, 0.5))

    }

    func addInstance(with geometry: Geometry, translation: SIMD3<Float> = .zero, rotation: SIMD3<Float> = .zero, scale: SIMD3<Float> = .one,
                     lightAmplifier: Float = 1.0, mask: UInt32 =  GEOMETRY_MASK_OPAQUE) {
        
        guard let lightGeometry = geometry.getLightGeometry() else { fatalError("Could not find light geometry") }
        
        if !geometry.vertices.isEmpty {
            geometry.uploadToBuffers()
            addGeometry(geometry)

            let opaqueInstance = GeometryInstance(geometry: geometry,
                                                  translation: translation,
                                                  rotation: rotation,
                                                  scale: scale,
                                                  mask: GEOMETRY_MASK_TRIANGLE | mask)
            instances.append(opaqueInstance)
        }

        if !lightGeometry.vertices.isEmpty {
            lightGeometry.lightAmplifier = lightAmplifier
            lightGeometry.uploadToBuffers()
            addGeometry(lightGeometry)
            
            let lightInstance = GeometryInstance(geometry: lightGeometry,
                                               translation: translation,
                                               rotation: rotation,
                                               scale: scale,
                                               mask: GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_OPAQUE)
            
            addLight(lightInstance: lightInstance, lightForward: SIMD3<Float>(0.0, -1.0, 0.0)) // TODO: do something with forward later
            instances.append(lightInstance)
        }
    }
    
    func addLight(lightInstance: GeometryInstance, lightForward: SIMD3<Float>) {
        let (newLightTriangles, newLightArea, averageColor) = lightInstance.getLightTriangles()
        
        let lightRight = simd_length(cross(worldUp, lightForward)) < 1e-3 ? SIMD3<Float>(1.0, 0.0, 0.0) : cross(worldUp, lightForward)
        let lightUp = cross(lightForward, lightRight)
        
        let leftAreaLight = AreaLight(position: lightInstance.translation,
                                      forward: simd_normalize(lightForward),
                                      right: simd_normalize(lightRight),
                                      up: simd_normalize(lightUp),
                                      color: averageColor,
                                      firstTriangleIndex: UInt32(lightTriangles.count),
                                      triangleCount: UInt32(newLightTriangles.count),
                                      totalArea: newLightArea,
                                      transform: lightInstance.transform
                                     )
        
        lightTriangles += newLightTriangles
        addLight(leftAreaLight)
    }
}
