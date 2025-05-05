//
//  CameraMovement.swift
//  PathTracing
//
//  Created on 5/4/25.
//

import MetalKit

class KeyHandlingMTKView: MTKView {
    weak var renderer: Renderer?
    var keysPressed = Set<UInt16>()
    
    override var acceptsFirstResponder: Bool {
        return true
    }
        
    override func keyDown(with event: NSEvent) {
        keysPressed.insert(event.keyCode)
        renderer?.keysPressed = keysPressed
    }
    
    override func keyUp(with event: NSEvent) {
        keysPressed.remove(event.keyCode)
        renderer?.keysPressed = keysPressed
    }
}

extension GameScene {
    func moveCamera(amount: Float, axis: SIMD3<Float>) {
        cameraPosition += axis * amount
        cameraTarget += axis * amount
    }
            
    func rotateUpDown(angle: Float) {
        let forward = cameraTarget - cameraPosition
        let distance = simd_length(forward)
        let direction = simd_normalize(forward)
        let right = simd_normalize(simd_cross(direction, cameraUp))
        
        let rotationMatrix = LinearAlgebra.rotateAroundAxis(-angle, around: normalize(right))
        let newDirection = simd_normalize(rotationMatrix * direction)
        
        cameraTarget = cameraPosition + newDirection * distance
    }
    
    func rotateLeftRight(angle: Float) {
        let forward = cameraTarget - cameraPosition
        let distance = simd_length(forward)
        let direction = simd_normalize(forward)
        
        let rotationMatrix = LinearAlgebra.rotateAroundAxis(-angle, around: cameraUp)
        let newDirection = simd_normalize(rotationMatrix * direction)
        
        cameraTarget = cameraPosition + newDirection * distance
    }
}

extension Renderer {
    func processCameraInput() {
        let wKey: UInt16 = 13
        let aKey: UInt16 = 0
        let sKey: UInt16 = 1
        let dKey: UInt16 = 2
        let lKey: UInt16 = 37
        let upArrow: UInt16 = 126
        let downArrow: UInt16 = 125
        let leftArrow: UInt16 = 123
        let rightArrow: UInt16 = 124
        let space: UInt16 = 49

        if keysPressed.isEmpty { return }
        
        let spacePressed = keysPressed.contains(space)
        var cameraChanged = false
        
        let forwardAxis = spacePressed ? simd_normalize(scene.cameraTarget - scene.cameraPosition) : worldForward
        let rightAxis = simd_normalize(simd_cross(forwardAxis, scene.cameraUp))

        if keysPressed.contains(wKey) {
            scene.moveCamera(amount: scene.cameraSpeed, axis: forwardAxis)
            cameraChanged = true
        }
        if keysPressed.contains(sKey) {
            scene.moveCamera(amount: -scene.cameraSpeed, axis: forwardAxis)
            cameraChanged = true
        }
        if keysPressed.contains(dKey) {
            scene.moveCamera(amount: scene.cameraSpeed, axis: rightAxis)
            cameraChanged = true
        }
        if keysPressed.contains(aKey) {
            scene.moveCamera(amount: -scene.cameraSpeed, axis: rightAxis)
            cameraChanged = true
        }
        
        if keysPressed.contains(upArrow) {
            scene.rotateUpDown(angle: scene.rotationSpeed)
            cameraChanged = true
        }
        if keysPressed.contains(downArrow) {
            scene.rotateUpDown(angle: -scene.rotationSpeed)
            cameraChanged = true
        }
        if keysPressed.contains(leftArrow) {
            scene.rotateLeftRight(angle: scene.rotationSpeed)
            cameraChanged = true
        }
        if keysPressed.contains(rightArrow) {
            scene.rotateLeftRight(angle: -scene.rotationSpeed)
            cameraChanged = true
        }
        
        if keysPressed.contains(lKey) {
            print("Camera Position: \(scene.cameraPosition)\nCamera Target: \(scene.cameraTarget)")
        }
        
        if cameraChanged {
            frameIndex = 0
        }
    }
}
