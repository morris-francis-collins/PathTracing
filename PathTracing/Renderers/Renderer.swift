//
//  Renderer.swift
//  PathTracing
//

import Foundation
import Metal
import MetalKit
import simd
import SwiftUI

let maxFramesInFlight: Int = 3
let alignedUniformsSize: Int = (MemoryLayout<Uniforms>.size + 255) & ~255

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    
    var copyPipeline: MTLRenderPipelineState!
    
    var uniformBuffer: MTLBuffer!
    
    var instanceAccelerationStructure: MTLAccelerationStructure?
    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
        
    var finalImage: MTLTexture?
    var randomTexture: MTLTexture!
    
    var textureArgumentBuffer: MTLBuffer!
    var textureCount: Int = 0
    var textureArray: MTLTexture!
    
    var resourceBuffer: MTLBuffer!
    var instanceBuffer: MTLBuffer!
    
    var visibleFunctionTable: MTLVisibleFunctionTable?
    
    let semaphore = DispatchSemaphore(value: maxFramesInFlight)
    var drawableSize: CGSize = .zero
    var uniformBufferOffset: Int = 0
    var uniformBufferIndex: Int = 0
    
    var frameIndex: UInt = 0
    var bufferPixels: Int = 0
    
    var scene: GameScene
    var keysPressed = Set<UInt16>()
    
    init(device: MTLDevice, scene: GameScene) {
        self.device = device
        self.scene = scene
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue.")
        }
        self.queue = queue
        
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to create default library.")
        }
        self.library = library
        
        super.init()
        
        createPipelines()
        createBuffers()
        createAccelerationStructures()
    }
    
    private func createPipelines() {
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        do {
            copyPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        } catch {
            fatalError("Failed to create render pipeline state: \(error)")
        }
    }
    
    func newComputePipelineState(function: MTLFunction) -> MTLComputePipelineState {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        
        do {
            let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            return pipeline
        } catch {
            fatalError("Failed to create compute pipeline state for function \(function.name): \(error)")
        }
    }
    
    func specializedFunction(named name: String) -> MTLFunction {
        guard let function = library.makeFunction(name: name) else {
            fatalError("Could not create specialized function named \(name)")
        }
        return function
    }
    
    func createBuffers() {
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight
        let options: MTLResourceOptions = getManagedBufferStorageMode()
        uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: options)
        
        scene.uploadToBuffers()
                
        createTextureArgumentBuffer()
    }
    
    func createTextureArgumentBuffer() {
        let textures = TextureRegistry.shared.getTextures()
        textureCount = textures.count
        
        var argumentDescriptors: [MTLArgumentDescriptor] = []
        
        let desc = MTLArgumentDescriptor()
        desc.index = 0
        desc.dataType = .texture
        desc.textureType = .type2D
        desc.arrayLength = Int(MAX_TEXTURES)
        desc.access = .readOnly
        argumentDescriptors.append(desc)
        
        guard let encoder = device.makeArgumentEncoder(arguments: argumentDescriptors) else {
            fatalError("Failed to create texture argument encoder")
        }
        
        let length = encoder.encodedLength
        textureArgumentBuffer = device.makeBuffer(
            length: length,
            options: .storageModeShared
        )
        
        encoder.setArgumentBuffer(textureArgumentBuffer, offset: 0)
        
        for i in 0..<Int(MAX_TEXTURES) {
            if i < textures.count {
                encoder.setTexture(textures[i], index: i)
            } else {
                encoder.setTexture(nil, index: i)
            }
        }
    }
    
    func newAccelerationStructure(descriptor: MTLAccelerationStructureDescriptor) -> MTLAccelerationStructure {
        let accelSizes = device.accelerationStructureSizes(descriptor: descriptor)
        let accelerationStructure = device.makeAccelerationStructure(size: accelSizes.accelerationStructureSize)!
        
        let scratchBuffer = device.makeBuffer(length: accelSizes.buildScratchBufferSize, options: .storageModePrivate)!
        
        guard let commandBuffer = queue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeAccelerationStructureCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder for acceleration structure build.")
        }
        
        let compactedSizeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        
        commandEncoder.build(accelerationStructure: accelerationStructure,
                             descriptor: descriptor,
                             scratchBuffer: scratchBuffer,
                             scratchBufferOffset: 0)
        
        commandEncoder.writeCompactedSize(accelerationStructure: accelerationStructure,
                                          buffer: compactedSizeBuffer,
                                          offset: 0)
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let compactedSize = compactedSizeBuffer.contents().assumingMemoryBound(to: UInt32.self).pointee
        let compactedAccelerationStructure = device.makeAccelerationStructure(size: Int(compactedSize))!
        
        guard let commandBuffer2 = queue.makeCommandBuffer(),
              let commandEncoder2 = commandBuffer2.makeAccelerationStructureCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder for compaction.")
        }
        commandEncoder2.copyAndCompact(sourceAccelerationStructure: accelerationStructure,
                                       destinationAccelerationStructure: compactedAccelerationStructure)
        commandEncoder2.endEncoding()
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()
        
        return compactedAccelerationStructure
    }
    
    func createAccelerationStructures() {
        let options: MTLResourceOptions = getManagedBufferStorageMode()
        primitiveAccelerationStructures = []
        
        for (i, geometry) in scene.geometries.enumerated() {
            if let geometryDescriptor = geometry.geometryDescriptor() {
                geometryDescriptor.intersectionFunctionTableOffset = i
                let accelDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
                accelDescriptor.geometryDescriptors = [geometryDescriptor]
                let accelStructure = newAccelerationStructure(descriptor: accelDescriptor)
                primitiveAccelerationStructures.append(accelStructure)
            } else {
                print("Warning: Failed to create geometry descriptor for geometry at index \(i)")
            }
        }
        
        let instanceDescriptorCount = scene.instances.count
        instanceBuffer = device.makeBuffer(length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * instanceDescriptorCount, options: options)!
        let instanceDescriptors = instanceBuffer.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: instanceDescriptorCount)
        
        for (instanceIndex, instance) in scene.instances.enumerated() {
            let geometryIndex = scene.geometries.firstIndex { $0 === instance.geometry } ?? 0
            instanceDescriptors[instanceIndex].accelerationStructureIndex = UInt32(geometryIndex)
            instanceDescriptors[instanceIndex].options = MTLAccelerationStructureInstanceOptions(rawValue: MTLAccelerationStructureInstanceOptions.opaque.rawValue)
            instanceDescriptors[instanceIndex].intersectionFunctionTableOffset = 0
            instanceDescriptors[instanceIndex].mask = UInt32(instance.mask)
            instanceDescriptors[instanceIndex].transformationMatrix = instance.getPackedTransform()
        }
        
        let accelDescriptor = MTLInstanceAccelerationStructureDescriptor()
        accelDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        accelDescriptor.instanceCount = instanceDescriptorCount
        accelDescriptor.instanceDescriptorBuffer = instanceBuffer
        accelDescriptor.usage = .preferFastIntersection
        
        instanceAccelerationStructure = newAccelerationStructure(descriptor: accelDescriptor)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
        let textureDescriptor = getImageTextureDescriptor(size)
        
        textureDescriptor.storageMode = .shared
        finalImage = device.makeTexture(descriptor: textureDescriptor)
                
        textureDescriptor.pixelFormat = .r32Uint
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.storageMode = .shared // ?
        
        randomTexture = device.makeTexture(descriptor: textureDescriptor)
        
        let pixelCount = Int(size.width * size.height)
        var randomValues = [UInt32](repeating: 0, count: pixelCount)
        for i in 0..<pixelCount {
            randomValues[i] = UInt32(arc4random_uniform(1024 * 1024))
        }

        randomValues.withUnsafeBytes { ptr in
            randomTexture.replace(region: MTLRegionMake2D(0, 0, Int(size.width), Int(size.height)),
                                  mipmapLevel: 0,
                                  withBytes: ptr.baseAddress!,
                                  bytesPerRow: MemoryLayout<UInt32>.size * Int(size.width))
        }
        
        frameIndex = 0
    }
    
    func updateUniforms() {
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        let uniformsPointer = uniformBuffer.contents().advanced(by: uniformBufferOffset).bindMemory(to: Uniforms.self, capacity: 1)
        var uniforms = uniformsPointer.pointee
        
        let position = scene.cameraPosition
        let target = scene.cameraTarget
        let up = scene.cameraUp
        
        let forward = simd_normalize(target - position)
        let right = simd_normalize(simd_cross(forward, up))
        let correctedUp = simd_normalize(simd_cross(right, forward))
        
        uniforms.camera.position = position
        uniforms.camera.forward = forward
        uniforms.camera.right = right
        uniforms.camera.up = correctedUp
        
        let fieldOfView: Float = Float(CAMERA_FOV_ANGLE) * (.pi / 180.0)
        let aspectRatio = Float(drawableSize.width / drawableSize.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight
        
        uniforms.camera.right *= imagePlaneWidth
        uniforms.camera.up *= imagePlaneHeight
        
        uniforms.width = UInt32(drawableSize.width)
        uniforms.height = UInt32(drawableSize.height)
        uniforms.frameIndex = UInt32(frameIndex)
        frameIndex += 1
        
        uniforms.lightCount = UInt32(scene.lights.count)
        uniformsPointer.pointee = uniforms
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
        
    func draw(in view: MTKView) {
        fatalError("Draw called on base Renderer")
    }
}

func getManagedBufferStorageMode() -> MTLResourceOptions {
#if os(iOS)
    return []
#else
    return .storageModeShared
#endif
}

func getImageTextureDescriptor(_ size: CGSize) -> MTLTextureDescriptor {
    let textureDescriptor = MTLTextureDescriptor()
    textureDescriptor.pixelFormat = .rgba32Float
    textureDescriptor.textureType = .type2D
    textureDescriptor.width = Int(size.width)
    textureDescriptor.height = Int(size.height)
    textureDescriptor.storageMode = .private
    textureDescriptor.usage = [.shaderRead, .shaderWrite]
    return textureDescriptor
}
