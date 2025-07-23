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
    var rotationSpeed: Float = 0.1
    
    var cameraLocations: [(SIMD3<Float>, SIMD3<Float>)] = []
    
    init(device: MTLDevice) {
        self.device = device
        createScene()
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
    
    func buildColorfulBox() { // x coordinates are flipped; left is positive, right is negative
        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL, textureURL: checkerBoardURL)
        wallGeometry.uploadToBuffers()
        addGeometry(wallGeometry)
        
        let whiteCubeGeometry = ObjGeometry(device: device, objURL: cubeURL)
        whiteCubeGeometry.uploadToBuffers()
        addGeometry(whiteCubeGeometry)
            
        var wallMaterial = PLASTIC
        
        wallMaterial.color = SIMD3<Float>(1.0, 0.0, 0.0);
        let redCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, material: wallMaterial)
        redCubeGeometry.uploadToBuffers()
        addGeometry(redCubeGeometry)

        wallMaterial.color = SIMD3<Float>(0.0, 1.0, 0.0);
        let greenCubeGeometry = ObjGeometry(device: device, objURL: cubeURL, material: wallMaterial)
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
            geometry: wallGeometry,
            translation: SIMD3<Float>(2.2, 1.8, 0.0),
            rotation: SIMD3<Float>(0, .pi/2, .pi/2),
            scale: SIMD3<Float>(4.0, 4.0, 0.4),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        let rightWallInstance = GeometryInstance(
            geometry: wallGeometry,
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
        
        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL, color: 0.6 * .one)
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
        
        let frontWallInstance = GeometryInstance(
            geometry: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, depth / 2),
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
        addInstance(frontWallInstance)
        addInstance(ceilingInstance)
        addInstance(middleWallInstance)
        addInstance(rightMirrorInstance)
        addInstance(leftMirrorInstance)
        
        buildWindowedWall(center: SIMD3<Float>(5, 5, 5), wallDimensions: SIMD2<Float>(7, 4), windowDimensions: SIMD2<Float>(3, 2))
        
        if (!extra) { return }
        
        let leftSideMirrorInstance = GeometryInstance(geometry: mirrorCubeGeometry,
                                                  translation: SIMD3<Float>(0.9, 2, 0),
                                                  scale: SIMD3<Float>(epsilon, 1, 1),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let rightSideMirrorInstance1 = GeometryInstance(geometry: mirrorCubeGeometry,
                                                        translation: SIMD3<Float>(1.1, 2, -2.25),
                                                        rotation: SIMD3<Float>(0, .pi, 0),
                                                        scale: SIMD3<Float>(epsilon, 4, 1.5),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        let rightSideMirrorInstance2 = GeometryInstance(geometry: mirrorCubeGeometry,
                                                        translation: SIMD3<Float>(1.1, 2, -0.25),
                                                        rotation: SIMD3<Float>(0, .pi, 0),
                                                        scale: SIMD3<Float>(epsilon, 4, 1.5),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)

        addInstance(leftSideMirrorInstance)
        addInstance(rightSideMirrorInstance1)
        addInstance(rightSideMirrorInstance2)
    }
    
    func buildBox(width: Float = 6.0, height: Float = 4.0, depth: Float = 6.0) {
        let epsilon: Float = 1e-3
        
        let wallGeometry = ModelIOGeometry(device: device, modelURL: cubeURL, defaultColor: 0.5 * .one, defaultMaterial: PLASTIC)
        let checkeredWallGeometry = ModelIOGeometry(device: device, modelURL: cubeURL, textureURL: checkerBoardURL, defaultMaterial: PLASTIC)
                        
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

        addInstance(
//            with: checkeredWallGeometry,
            with: wallGeometry,
            translation: SIMD3<Float>(-width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(width / 2, height / 2, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(epsilon, height, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
                
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, epsilon, depth),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, height / 2, depth / 2),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(width, height, epsilon),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )

//        addInstance(floorInstance)
//        addInstance(leftWallInstance)
//        addInstance(rightWallInstance)
//        addInstance(backWallInstance)
//        addInstance(ceilingInstance)
//        addInstance(frontWallInstance)
    }
    
    func buildWindowedBox(width: Float = 8.0, height: Float = 4.0, depth: Float = 8.0) {
        let epsilon: Float = 1e-3

        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL)
        wallGeometry.uploadToBuffers()
        addGeometry(wallGeometry)
        
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, 0.0, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, epsilon, depth)
        )
        
        buildWindowedWall(center: SIMD3<Float>(-width / 2, height / 2, 0.0),
                          wallDimensions: SIMD2<Float>(depth, height),
                          windowDimensions: SIMD2<Float>(2, 2),
                          rotation: SIMD3<Float>(0, .pi/2, 0)
        )
        
        buildWindowedWall(center: SIMD3<Float>(width / 2, height / 2, 0.0),
                          wallDimensions: SIMD2<Float>(depth, height),
                          windowDimensions: 0 * SIMD2<Float>(2, 2),
                          rotation: SIMD3<Float>(0, .pi/2, 0)
        )

        buildWindowedWall(center: SIMD3<Float>(0.0, height / 2, depth / 2),
                          wallDimensions: SIMD2<Float>(depth, height),
                          windowDimensions: 0 * SIMD2<Float>(2, 2),
                          rotation: SIMD3<Float>(0, 0, 0)
        )
        
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, height / 2, -depth / 2),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, height, epsilon)
        )
        
        addInstance(with: wallGeometry,
                    translation: SIMD3<Float>(0.0, height, 0.0),
                    rotation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(width, epsilon, depth)
        )
    }
    
    func buildWindowedWall(center: SIMD3<Float>, wallDimensions: SIMD2<Float>, windowDimensions: SIMD2<Float>, rotation: SIMD3<Float> = .zero) {
        let epsilon: Float = 1e-3
        
        let wallGeometry = ObjGeometry(device: device, objURL: cubeURL)
        wallGeometry.uploadToBuffers()
        addGeometry(wallGeometry)
        
        let glassGeometry = ObjGeometry(device: device, objURL: cubeURL, material: GLASS)
        glassGeometry.uploadToBuffers()
        addGeometry(glassGeometry)
        
        let qx = simd_quatf(angle: rotation.x, axis: SIMD3<Float>(1,0,0))
        let qy = simd_quatf(angle: rotation.y, axis: SIMD3<Float>(0,1,0))
        let qz = simd_quatf(angle: rotation.z, axis: SIMD3<Float>(0,0,1))
        let quat = qz * qy * qx
        
        let wW = wallDimensions.x
        let wH = wallDimensions.y
        let winW = windowDimensions.x
        let winH = windowDimensions.y
        let frameW = (wW - winW) * 0.5
        let frameH = (wH - winH) * 0.5
        
        func addRotated(geometry: ObjGeometry, offset: SIMD3<Float>, scale: SIMD3<Float>) {
            let rotatedOffset = quat.act(offset)
            
            addInstance(
                with: geometry,
                translation: center + rotatedOffset,
                rotation: rotation,
                scale: scale
            )
        }
        
        addRotated(
            geometry: glassGeometry,
            offset: .zero,
            scale: SIMD3<Float>(winW, winH, epsilon)
        )
        
        let leftOffset = SIMD3<Float>(-(winW + frameW) / 2, 0, 0)
        let rightOffset = SIMD3<Float>((winW + frameW) / 2, 0, 0)
        let bottomOffset = SIMD3<Float>(0, -(winH + frameH) / 2, 0)
        let topOffset = SIMD3<Float>(0, (winH + frameH) / 2, 0)
        
        addRotated(geometry: wallGeometry,
                   offset: leftOffset,
                   scale: SIMD3<Float>(frameW, wH, epsilon))
        
        addRotated(geometry: wallGeometry,
                   offset: rightOffset,
                   scale: SIMD3<Float>(frameW, wH, epsilon))
        
        addRotated(geometry: wallGeometry,
                   offset: bottomOffset,
                   scale: SIMD3<Float>(winW, frameH, epsilon))
        
        addRotated(geometry: wallGeometry,
                   offset: topOffset,
                   scale: SIMD3<Float>(winW, frameH, epsilon))
    }

    func createScene() { // x coordinates are flipped; left is positive, right is negative
        cameraPosition = SIMD3<Float>(0.0, 2.0, -5.0)
        cameraTarget = SIMD3<Float>(0.0, 1.75, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        buildColorfulBox()


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
        
        let torusGeometry = ObjGeometry(device: device, objURL: torusURL, material: GLASS)
        torusGeometry.uploadToBuffers()
        addGeometry(torusGeometry)

        let cubeInstance = GeometryInstance(
            geometry: whiteCubeGeometry,
            translation: SIMD3<Float>(-1.0, 0.2 + 0.4, -0.1),
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
            geometry: torusGeometry,
            translation: SIMD3<Float>(-1.2, 1.2, -0.5),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(1, 0.1, 1),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT
        )
        
        let leftBallInstance = GeometryInstance(
            geometry: torusGeometry,
            translation: SIMD3<Float>(1.2, 0.7, 0.8),
            rotation: SIMD3<Float>(.pi/4, 0, 0),
            scale: SIMD3<Float>(1.5, 0.15, 1.5),
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
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, .pi/4, 0),
            scale: SIMD3<Float>(2.0, 1.0, 1.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        var GOLD = GLASS
        GOLD.color = SIMD3<Float>(1.0, 0.84, 0.6)
        GOLD.roughness = 0.2
        let dragonGeometry = ModelIOGeometry(device: device, modelURL: dragonURL, defaultColor: .one, defaultMaterial: GOLD)
        let angelGeometry = ObjGeometry(device: device, objURL: angelURL, color: .one, material: GLASS)
        let saintGeometry = ModelIOGeometry(device: device, modelURL: saintURL, defaultColor: .one, defaultMaterial: GLASS)
        
//        addInstance(with: angelGeometry,
//                    translation: SIMD3<Float>(0, 0, 0),
//                    rotation: SIMD3<Float>(.pi, 0, 0),
//                    scale: SIMD3<Float>(0.0020, 0.0020, 0.0020)
//        )

//        addInstance(with: saintGeometry,
//                    translation: SIMD3<Float>(0, -5, 0),
//                    rotation: SIMD3<Float>(-.pi/2, 0, 0),
//                    scale: SIMD3<Float>(0.03, 0.03, 0.03)
//        )


        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(0.0, -0.5, 0.0),
                    rotation: SIMD3<Float>(0, .pi + 0.5, 0),
                    scale: SIMD3<Float>(0.15, 0.15, 0.15)
        )

//        addInstance(cubeInstance)
//        addInstance(ballInstance)
//        addInstance(rightBallInstance)
//        addInstance(leftBallInstance)
//        addInstance(pyramidInstance)
//        addInstance(tableInstance)
        
        let lightGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), emissionColor: 10 * .one, material: GLASS)

        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.399, -1.0),
                    scale: SIMD3<Float>(4.0, 0.00001, 0.8))
        
        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.399, 1.0),
                    scale: SIMD3<Float>(4.0, 0.00001, 0.8))
        
//        let lightBallGeometry = ModelIOGeometry(device: device, modelURL: ballURL, emissionColor: SIMD3<Float>(repeating: 10.0))
//        
//        addInstance(with: lightBallGeometry,
//                    translation: SIMD3<Float>(-1.0, 2.5, 0.0),
//                    scale: 1.0 * SIMD3<Float>(1.0, 1.0, 1.0)
//                      )

    }
    
    func createLivelyScene() {
        cameraLocations = [(SIMD3<Float>(1.208283, 2.3361523, 3.7958465), SIMD3<Float>(0.20828247, 1.836154, -2.2041554))
                          ]
        
//        cameraPosition = SIMD3<Float>(1, 2.5, 6)
//        cameraTarget = SIMD3<Float>(0.0, 2, 0.0)
        (cameraPosition, cameraTarget) = cameraLocations[0]
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
        
        let fluorescentLampGeometry = ModelIOGeometry(device: device, modelURL: fluorescentURL)
            
        let tableInstance = GeometryInstance(
            geometry: tableGeometry,
            translation: SIMD3<Float>(2.5, 0.0, -1.5),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(2.0, 1.5, 4.0),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        let glassBallInstance = GeometryInstance(
            geometry: glassBallGeometry,
            translation: SIMD3<Float>(2.5, 1.5 + 0.375, -0.3),
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
        
//        addInstance(with: hangingLightGeometry,
//                    translation: SIMD3<Float>(-0.5, -0.25, 2.0),
//                    rotation: SIMD3<Float>(0, .pi/2, 0),
//                    scale: SIMD3<Float>(0.005, 0.005, 0.005),
//                    lightAmplifier: 5.0,
//                    mask: GEOMETRY_MASK_OPAQUE)
        
//        addInstance(with: wallLightGeometry,
//                    translation: SIMD3<Float>(1.25, 2.5, 0.0),
//                    rotation: SIMD3<Float>(0, .pi/2 + 0.55, 0),
//                    scale: SIMD3<Float>(0.005, 0.005, 0.005),
//                    lightAmplifier: 3.0,
//                    mask: GEOMETRY_MASK_OPAQUE)
        
//        addInstance(with: floorLampGeometry,
//                    translation: SIMD3<Float>(-3.5, 0.0, -3.5),
//                    rotation: SIMD3<Float>(0, 0, 0),
//                    scale: SIMD3<Float>(0.015, 0.015, 0.015),
//                    lightAmplifier: 3.0,
//                    mask: GEOMETRY_MASK_OPAQUE)
//        
//        addInstance(with: fluorescentLampGeometry,
//                    translation: SIMD3<Float>(2.25, 5.45, -2.0),
//                    rotation: SIMD3<Float>(.pi, 0, 0),
//                    scale: SIMD3<Float>(0.01, 0.01, 0.01),
//                    lightAmplifier: 20.0,
//                    mask: GEOMETRY_MASK_OPAQUE)

        
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 5))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(-0.5, 1.75, -2.75),
                    scale: 1 * SIMD3<Float>(0.5, 0.5, 0.5))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(2.5, 1.5 + 0.375 + 1.5, -1.0),
                    scale: 1 * SIMD3<Float>(0.5, 0.5, 0.5))
        
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(-0.5, 1.75, 1.0),
                    scale: 1 * SIMD3<Float>(0.5, 0.5, 0.5))

                        

        addInstance(tableInstance)
//        addInstance(glassBallInstance)
//        addInstance(glassrightBallInstance)
        addInstance(mirrorBallInstance)
        addInstance(mirrorInstance)
        addInstance(teaTableInstance)
        addInstance(couchInstance)
    }
            
    func createDifficultScene() {
        cameraPosition = SIMD3<Float>(0, 2.5, 6)
        cameraTarget = SIMD3<Float>(0, 2, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        cameraLocations = [(cameraPosition, cameraTarget),
                            (SIMD3<Float>(2.9280746, 2.5814812, 3.0649862), SIMD3<Float>(-0.489712, 1.8794974, -1.8417473)),
                            (SIMD3<Float>(3.860585, 2.7261002, -0.094045304), SIMD3<Float>(-0.15811086, -0.75679064, -2.9171062)),
                            (SIMD3<Float>(3.860585, 1.2261002, 0.9059547), SIMD3<Float>(-0.120718956, -1.2116656, -2.896234)),
                            (SIMD3<Float>(-1.139415, 2.2261002, -0.094045304), SIMD3<Float>(4.7482796, 1.22613, -0.85901386))
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
                                                 translation: SIMD3<Float>(2.5, 0.75, -0.25),
                                                 scale: SIMD3<Float>(1.5, 1.5, 1.5),
                                                 mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
        
        let plasticBallInstance = GeometryInstance(geometry: plasticBallGeometry,
                                                   translation: SIMD3<Float>(2.5, 0.75, -0.25),
                                                   scale: SIMD3<Float>(0.55, 0.55, 0.55),
                                                   mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let sideMirrorInstance = GeometryInstance(geometry: mirrorCubeGeometry,
                                                  translation: SIMD3<Float>(0.9, 2, 0),
                                                  rotation: SIMD3<Float>(0, 0, 0),
                                                  scale: SIMD3<Float>(0.001, 1, 1),
                                                  mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE)
        
        let frontRightGlassBallInstance = GeometryInstance(geometry: glassBallGeometry,
                                                           translation: SIMD3<Float>(2.5, 0.75, 2.0),
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
        
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 5.0))

        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(0.0, 2.5, -3), scale: SIMD3<Float>(0.5, 0.5, 0.5))
        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(2.0, 2.5, -3), scale: SIMD3<Float>(0.5, 0.5, 0.5))
    }
    
    func createFocusScene() {
        cameraPosition = SIMD3<Float>(0.0, 2.0, 2.5)
        cameraTarget = SIMD3<Float>(0.0, 1.75, 0.0)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildBox()
        
        cameraLocations = [(cameraPosition, cameraTarget),
                           (SIMD3<Float>(-1.4804919, 1.75, -1.4395511), SIMD3<Float>(0.55699587, 1.1292119, -0.1069715)),
                           (SIMD3<Float>(-0.06333873, 1.7002482, 1.6584291), SIMD3<Float>(0.4302318, 1.3256127, -0.77642965)),
                           (SIMD3<Float>(-0.66476583, 1.564688, -1.1811823), SIMD3<Float>(0.69295394, 0.94389987, 0.83963954)),
                           (SIMD3<Float>(-0.40038502, 1.5263345, -1.2669994), SIMD3<Float>(0.07269779, 1.1516991, 1.1719233)),
                           (SIMD3<Float>(0.0, 1.7512412, 0.012407258), SIMD3<Float>(-1.9680835, 1.7520738, -1.5493672)),
                           (SIMD3<Float>(-0.5884206, 1.7529173, -1.1681728), SIMD3<Float>(0.22924161, 0.44039154, 0.8120316))
                          ]
        
        let porscheGeometry = ModelIOGeometry(device: device, modelURL: porscheURL, inwardsNormals: false)
        addInstance(with: porscheGeometry,
                    translation: SIMD3<Float>(0.0, 1.5, 0.0),
                    rotation: SIMD3<Float>(0.0, .pi/6, 0.0),
                    scale: SIMD3<Float>(0.005, 0.005, 0.005)
                    )
        
        let arrowGeometry = ModelIOGeometry(device: device, modelURL: arrowURL)
        addInstance(with: arrowGeometry,
                    translation: SIMD3<Float>(0.0, 2.5, 0.0),
                    rotation: SIMD3<Float>(0.0, -.pi/6, 0.0),
                    scale: SIMD3<Float>(0.05, 0.05, 0.05)
                    )
                
//        let bugattiGeometry = ModelIOGeometry(device: device, modelURL: bugattiURL)
//        addInstance(with: bugattiGeometry,
//                    translation: SIMD3<Float>(0.0, 1.0, 0.0),
//                    rotation: SIMD3<Float>(0.0, -.pi/6, 0.0),
//                    scale: SIMD3<Float>(0.005, 0.005, 0.005)
//                    )
        
//        let futuristicCarGeometry = ModelIOGeometry(device: device, modelURL: futuristicCarURL)
//        addInstance(with: futuristicCarGeometry,
//                    translation: SIMD3<Float>(0.0, 0.0, -1.5),
//                    rotation: SIMD3<Float>(0.0, .pi/5, 0.0),
//                    scale: SIMD3<Float>(0.005, 0.005, 0.005)
//                    )

//        let miniCooperGeometry = ModelIOGeometry(device: device, modelURL: miniCooperURL)
//        addInstance(with: miniCooperGeometry,
//                    translation: SIMD3<Float>(0.0, 1.0, 0.0),
//                    rotation: SIMD3<Float>(0.0, -.pi/6, 0.0),
//                    scale: SIMD3<Float>(0.005, 0.005, 0.005)
//                    )
        
//        let waterGeometry = ObjGeometry(device: device, objURL: waterURL, color: SIMD3<Float>(1, 1, 1), material: GLASS)
//        waterGeometry.uploadToBuffers()
//        addGeometry(waterGeometry)
//        
//        let waterInstance = GeometryInstance(geometry: waterGeometry,
//                                                 translation: SIMD3<Float>(0, 2, 0),
//                                                 rotation: SIMD3<Float>(0, 0, 0),
//                                             scale: SIMD3<Float>(1.0, 4.0, 1.0),
//                                                 mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_TRANSPARENT)
//
//        addInstance(waterInstance)
        
//        let M4A1Geometry = ModelIOGeometry(device: device, modelURL: M4A1PrintStreamURL)
//        addInstance(with: M4A1Geometry,
//                    translation: SIMD3<Float>(0.0, 2.0, 0.0),
//                    rotation: SIMD3<Float>(0.0, -.pi/6, 0.0),
//                    scale: SIMD3<Float>(0.003, 0.003, 0.003)
//                    )

        let lightGeometry = ObjGeometry(device: device, objURL: cubeURL, color: .one, emissionColor: 5 * .one)
                
        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.99, -1.0),
                    scale: SIMD3<Float>(3.9, 0.00001, 0.8)
                    )
        
        addInstance(with: lightGeometry,
                    translation: SIMD3<Float>(0.0, 3.99, 1.0),
                    scale: SIMD3<Float>(3.9, 0.00001, 0.8)
                    )
    }
    
    func createLivingRoomScene() {
        cameraPosition = SIMD3<Float>(59.49535, 35.642956, 33.0719)
        cameraTarget = SIMD3<Float>(58.55864, 35.40372, 32.816353)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        let livingRoomGeometry = ModelIOGeometry(device: device, modelURL: whiteLivingRoomURL)
        
        addInstance(with: livingRoomGeometry,
                    translation: SIMD3<Float>(0.0, 0, 0.0),
                    scale: SIMD3<Float>(0.2, 0.2, 0.2),
                    lightAmplifier: 2.0
                    )
                
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 50.0))
        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(-32.878624, 29.477142, -37.870895), scale: SIMD3<Float>(2, 2, 2))
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
        
        (cameraPosition, cameraTarget) = cameraLocations[5]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        buildBox(width: 6.0, height: 4.0, depth: 6.0)
        
        let mirrorWallGeometry = ObjGeometry(device: device, objURL: cubeURL, color: SIMD3<Float>(1, 1, 1), material: MIRROR)
        let dragonGeometry = ModelIOGeometry(device: device, modelURL: dragonURL, defaultColor: .one, defaultMaterial: GLASS)
        let angelGeometry = ObjGeometry(device: device, objURL: angelURL, color: .one, material: GLASS)
        let saintGeometry = ModelIOGeometry(device: device, modelURL: saintURL, defaultColor: .one, defaultMaterial: GLASS)


        addInstance(with: mirrorWallGeometry,
                    translation: SIMD3<Float>(-1.0, 1.5, -2.95),
                    scale: SIMD3<Float>(2.0, 2.0, 0.01)
        )
        
        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(-1.8, 0.0, 2.5),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(0.1, 0.1, 0.1)
        )
//        
        addInstance(with: dragonGeometry,
                    translation: SIMD3<Float>(-1.8, 2.0, 2.5),
                    rotation: SIMD3<Float>(0, .pi, 0),
                    scale: SIMD3<Float>(1, 1, 1)
        )
//
//        
//        addInstance(with: angelGeometry,
//                    translation: SIMD3<Float>(0, 0, 2.5),
//                    rotation: SIMD3<Float>(.pi, 0, 0),
//                    scale: SIMD3<Float>(0.0015, 0.0015, 0.0015)
//        )
//        
//        addInstance(with: angelGeometry,
//                    translation: SIMD3<Float>(0, 5.0, 10.0),
//                    rotation: SIMD3<Float>(.pi, 0, 0),
//                    scale: SIMD3<Float>(0.01, 0.01, 0.01)
//        )
//
//        
//        addInstance(with: saintGeometry,
//                    translation: SIMD3<Float>(1.5, 0, 2.2),
//                    rotation: SIMD3<Float>(-.pi/2, 0, 0),
//                    scale: SIMD3<Float>(0.01, 0.01, 0.01)
//        )
//        
//        addInstance(with: saintGeometry,
//                    translation: SIMD3<Float>(1.5, 4.0, -5.0),
//                    rotation: SIMD3<Float>(-.pi/2, 0, 0),
//                    scale: SIMD3<Float>(0.1, 0.1, 0.1)
//        )

        
        var plasticMaterial = PLASTIC
        
        for i in 0...10 {
            plasticMaterial.roughness = Float(i) * 0.1
            
            let scalingPlasticGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: plasticMaterial)

            addInstance(with: scalingPlasticGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, -1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }

        var mirrorMaterial = MIRROR
        
        for i in 0...10 {
            mirrorMaterial.roughness = Float(i) * 0.1
//            mirrorMaterial.color = (1.0 - colors[i])

            let scalingMirrorGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: mirrorMaterial)

            addInstance(with: scalingMirrorGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 0.0),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
        
//        for i in 0...10 {
//            mirrorMaterial.roughness_x = Float(i) * 0.1
//            mirrorMaterial.roughness_y = 0.1
//
//            let scalingMirrorGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: mirrorMaterial)
//            
//            addInstance(with: scalingMirrorGeometry,
//                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 0.5),
//                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
//            )
//        }
        
        var glassMaterial = GLASS
        
        for i in 0...10 {
            glassMaterial.roughness = Float(i) * 0.1

            let scalingGlassGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: glassMaterial)

            addInstance(with: scalingGlassGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }

        var coloredGlassMaterial = GLASS
        
        for i in 0...10 {
            coloredGlassMaterial.color = (colors[i])
            
            let coloredGlassGeometry = ModelIOGeometry(device: device, modelURL: cubeURL, defaultColor: colors[i], defaultMaterial: coloredGlassMaterial)
            
            addInstance(with: coloredGlassGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 2.0, 2.8),
                        scale: SIMD3<Float>(0.4, 0.1, 0.4)
                        )
        }

        
        let lightBallGeometry = ModelIOGeometry(device: device, modelURL: ballURL, emissionColor: SIMD3<Float>(repeating: 10.0))
        let lightStripGeometry = ModelIOGeometry(device: device, modelURL: cubeURL, emissionColor: SIMD3<Float>(repeating: 5.0))
                                
//        addInstance(with: lightStripGeometry,
//                    translation: SIMD3<Float>(0.0, 3.99, 0.0),
//                    scale: SIMD3<Float>(3.9, 0.00001, 0.8)
//                    )
        
//        addInstance(with: lightStripGeometry,
//                    translation: SIMD3<Float>(0.0, 3.99, -2.0),
//                    scale: SIMD3<Float>(3.9, 0.00001, 0.8)
//                    )
//
//        addInstance(with: lightStripGeometry,
//                    translation: SIMD3<Float>(0.0, 3.99, 2.0),
//                    scale: SIMD3<Float>(3.9, 0.00001, 0.8)
//                    )
        
//        addInstance(with: lightBallGeometry,
//                    translation: SIMD3<Float>(-0.0, 3.99 - 1, 0.0),
//                    scale: 10.0 * SIMD3<Float>(0.5, 0.001, 0.5)
//                      )
//        
//        addInstance(with: lightBallGeometry,
//                    translation: SIMD3<Float>(-0.0, 3.99, 0.0),
//                    scale: 5 * SIMD3<Float>(0.5, 0.001, 0.5)
//                      )
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(0.0, 2.5, 0.0),
                    scale: 1.0 * SIMD3<Float>(1.0, 1.0, 1.0)
                      )
    }
    
    func createWaterScene() {
        cameraLocations = [(SIMD3<Float>(0.0, 0.5, -2.0), SIMD3<Float>(0.0, 0.5, -1.0))
                          ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        let width: Float = 10.0
        let height: Float = 5.0
        let depth: Float = 10.0
        
        buildBox(width: 10.0, height: 5.0, depth: 10.0)
        
        let waterGeometry = ObjGeometry(device: device, objURL: waterURL, material: WATER)
        
        addInstance(with: waterGeometry,
                    translation: SIMD3<Float>(0, 0, 0),
                    scale: SIMD3<Float>(20 * 1.55, 3.5, 20 * 1.55)
        )
        
        let lightBallGeometry = ModelIOGeometry(device: device, modelURL: ballURL, emissionColor: SIMD3<Float>(repeating: 3.0))
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(-0.0, 3.99, 0.0),
                    scale: 5.0 * SIMD3<Float>(0.5, 0.001, 0.5)
                      )
    }
    
    func createWindowScene() {
        cameraLocations = [(SIMD3<Float>(0.0, 0.5, -2.0), SIMD3<Float>(0.0, 0.5, 1.0)),
                           (SIMD3<Float>(1.5124673, 2.0426145, -2.9677484), SIMD3<Float>(0.63033533, 1.7431145, -0.11605692)),
                           (SIMD3<Float>(1.5124673, 0.54261446, -2.9677484), SIMD3<Float>(0.63033533, 0.24311447, -0.11605692))
                          ]
        
        (cameraPosition, cameraTarget) = cameraLocations[1]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)
        
        buildBox(width: 6, height: 4, depth: 6)
        
//        let dragonGeometry = ModelIOGeometry(device: device, modelURL: dragonURL, defaultColor: .one, defaultMaterial: GLASS)
//        
//        addInstance(with: dragonGeometry,
//                    translation: SIMD3<Float>(-1.8, 0.0, 2.5),
//                    rotation: SIMD3<Float>(0, .pi, 0),
//                    scale: SIMD3<Float>(0.1, 0.1, 0.1)
//        )
        
        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 50))
        
        addInstance(with: lightBallGeometry,
                    translation: SIMD3<Float>(0, 0.5, 0),
                    scale: SIMD3<Float>(0.5, 0.5, 0.5))
    }
    
    func createBathroomScene() {
        cameraPosition = SIMD3<Float>(0, 0, 0)
        cameraTarget = SIMD3<Float>(0, 0, 1)
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        let bathroomGeometry = ModelIOGeometry(device: device, modelURL: bathroomURL)
        
        addInstance(with: bathroomGeometry,
                    translation: SIMD3<Float>(0.0, 0, 0.0),
                    scale: SIMD3<Float>(1, 1, 1)
                    )
                
//        let lightBallGeometry = ObjGeometry(device: device, objURL: ballURL, emissionColor: SIMD3<Float>(repeating: 50.0))
//        addInstance(with: lightBallGeometry, translation: SIMD3<Float>(-32.878624, 29.477142, -37.870895), scale: SIMD3<Float>(2, 2, 2))
    }
    
    func createWhiteFurnaceScene() {
        cameraLocations = [(SIMD3<Float>(0.0, 3.95, 0.0), SIMD3<Float>(0.0, 1.0, -0.01))
                          ]
        
        (cameraPosition, cameraTarget) = cameraLocations[0]
        cameraUp = SIMD3<Float>(0.0, 1.0, 0.0)

        let wallGeometry = ModelIOGeometry(device: device, modelURL: cubeURL, defaultColor: .one, defaultMaterial: PLASTIC)
                        
        addInstance(
            with: wallGeometry,
            translation: SIMD3<Float>(0.0, 0.0, 0.0),
            rotation: SIMD3<Float>(0, 0, 0),
            scale: SIMD3<Float>(6, 1e-3, 6),
            mask: GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_OPAQUE
        )
        
        var plasticMaterial = PLASTIC
        
        for i in 0...10 {
            plasticMaterial.roughness = Float(i) * 0.1
            
            let scalingPlasticGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: plasticMaterial)

            addInstance(with: scalingPlasticGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, -1.5),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }

        var mirrorMaterial = MIRROR
        
        for i in 0...10 {
            mirrorMaterial.roughness = Float(i) * 0.1

            let scalingMirrorGeometry = ModelIOGeometry(device: device, modelURL: ballURL, defaultMaterial: mirrorMaterial)

            addInstance(with: scalingMirrorGeometry,
                        translation: SIMD3<Float>(2.5 - 0.5 * Float(i), 0.2, 0.0),
                        scale: SIMD3<Float>(0.4, 0.4, 0.4)
            )
        }
        
        let lightBallGeometry = ModelIOGeometry(device: device, modelURL: ballURL, emissionColor: SIMD3<Float>(repeating: 5.0))
        
                addInstance(with: lightBallGeometry,
                            translation: SIMD3<Float>(-0.0, 3.99, 0.0),
                            scale: 5 * SIMD3<Float>(0.5, 0.001, 0.5)
                            )
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
                                               mask: GEOMETRY_MASK_LIGHT)
            
            addLight(lightInstance: lightInstance, lightForward: SIMD3<Float>(0.0, -1.0, 0.0)) // TODO: do something with forward later
            instances.append(lightInstance)
        }
    }
    
    func addLight(lightInstance: GeometryInstance, lightForward: SIMD3<Float>) {
        let (newLightTriangles, newLightArea, averageColor) = lightInstance.getLightTriangles()
        let leftAreaLight = AreaLight(position: lightInstance.translation,
                                      color: averageColor,
                                      firstTriangleIndex: UInt32(lightTriangles.count),
                                      triangleCount: UInt32(newLightTriangles.count),
                                      totalArea: newLightArea
                                     )
        
        lightTriangles += newLightTriangles
        addLight(leftAreaLight)
    }
}
