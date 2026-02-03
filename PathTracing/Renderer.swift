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
    
    var uniformBuffer: MTLBuffer!
    
    var instanceAccelerationStructure: MTLAccelerationStructure?
    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    
    var raytracingPipeline: MTLComputePipelineState!
    var copyPipeline: MTLRenderPipelineState!
    var clearBufferPipeline: MTLComputePipelineState!
    var finalizePipeline: MTLComputePipelineState!
    
    var finalImage: MTLTexture?
    var accumulationTargets: [MTLTexture?] = [nil, nil]
    var splatTargets: [MTLTexture?] = [nil, nil]
    var randomTexture: MTLTexture!
    
    var atomicSplatBuffer: MTLBuffer!
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
        
        createBuffers()
        createAccelerationStructures()
        createPipelines()
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
    
    func createPipelines() {
        let raytracingFunction = specializedFunction(named: "raytracingKernel")
        raytracingPipeline = newComputePipelineState(function: raytracingFunction)
                
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        do {
            copyPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        } catch {
            fatalError("Failed to create render pipeline state: \(error)")
        }
        
        let clearFunction = specializedFunction(named: "clearAtomicBuffer")
        clearBufferPipeline = newComputePipelineState(function: clearFunction)
        
        let finalizeFunction = specializedFunction(named: "finalizeAtomicBuffer")
        finalizePipeline = newComputePipelineState(function: finalizeFunction)
    }
        
    func createBuffers() {
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight
        let options: MTLResourceOptions = getManagedBufferStorageMode()
        uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: options)
        
        scene.uploadToBuffers()
                
        let bufferSize = Int(2 * PIXEL_WIDTH) * Int(2 * PIXEL_HEIGHT) * 3 * MemoryLayout<Float>.size
        atomicSplatBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        )
        
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
                encoder.setTexture(nil, index: i)  // Pad with nil
            }
        }
        
        //        #if !os(iOS)
        //        textureArgumentBuffer.didModifyRange(0..<length)
        //        #endif
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
            instanceDescriptors[instanceIndex].options = (instance.geometry.intersectionFunctionName() == nil)
            ? MTLAccelerationStructureInstanceOptions(rawValue: MTLAccelerationStructureInstanceOptions.opaque.rawValue)
            : []
            instanceDescriptors[instanceIndex].intersectionFunctionTableOffset = 0
            instanceDescriptors[instanceIndex].mask = UInt32(instance.mask)
            instanceDescriptors[instanceIndex].transformationMatrix = instance.getPackedTransform()
        }
        
        //        #if !os(iOS)
        //        instanceBuffer.didModifyRange(0..<instanceBuffer.length)
        //        #endif
        
        let accelDescriptor = MTLInstanceAccelerationStructureDescriptor()
        accelDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        accelDescriptor.instanceCount = instanceDescriptorCount
        accelDescriptor.instanceDescriptorBuffer = instanceBuffer
        accelDescriptor.usage = .preferFastIntersection
        
        instanceAccelerationStructure = newAccelerationStructure(descriptor: accelDescriptor)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = .rgba32Float
        textureDescriptor.textureType = .type2D
        textureDescriptor.width = Int(size.width)
        textureDescriptor.height = Int(size.height)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        finalImage = device.makeTexture(descriptor: textureDescriptor)
        
        for i in 0..<2 {
            accumulationTargets[i] = device.makeTexture(descriptor: textureDescriptor)
        }
        
        for i in 0..<2 {
            splatTargets[i] = device.makeTexture(descriptor: textureDescriptor)
        }
        
        textureDescriptor.pixelFormat = .r32Uint
        textureDescriptor.usage = [.shaderRead]
#if !os(iOS)
        textureDescriptor.storageMode = .shared // change????
#else
        textureDescriptor.storageMode = .shared
#endif
        
        randomTexture = device.makeTexture(descriptor: textureDescriptor)
        
        // random texture data
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
        
        //        #if !os(iOS)
        //        uniformBuffer.didModifyRange(uniformBufferOffset..<uniformBufferOffset + alignedUniformsSize)
        //        #endif
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
    
    func draw(in view: MTKView) {
        let t0 = CFAbsoluteTimeGetCurrent()
        
        _ = semaphore.wait(timeout: .distantFuture)
        let t1 = CFAbsoluteTimeGetCurrent()
                
        guard let commandBuffer = queue.makeCommandBuffer() else {
            return
        }

        commandBuffer.addCompletedHandler { _ in
            self.semaphore.signal()
        }
        
        let t2 = CFAbsoluteTimeGetCurrent()
        
        processCameraInput()
        let t3 = CFAbsoluteTimeGetCurrent()
        
        updateUniforms()
        let t4 = CFAbsoluteTimeGetCurrent()
        
        let width = Int(drawableSize.width)
        let height = Int(drawableSize.height)
        
        let threadWidth = raytracingPipeline.threadExecutionWidth
        let threadHeight = raytracingPipeline.maxTotalThreadsPerThreadgroup / threadWidth
        
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(width: (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                   height: (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                   depth: 1)
        
        // clearAtomicBuffer
        guard let clearEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        clearEncoder.setComputePipelineState(clearBufferPipeline)
        clearEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 0)
        clearEncoder.setBytes([UInt32(width), UInt32(height)], length: 8, index: 1)
        clearEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        clearEncoder.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        let t5 = CFAbsoluteTimeGetCurrent()
        
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
//        computeEncoder.setBuffer(resourceBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(instanceBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(scene.lightBuffer, offset: 0, index: 2)
        computeEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: 3)
        computeEncoder.setVisibleFunctionTable(visibleFunctionTable, bufferIndex: 4)
        computeEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 5)
        computeEncoder.setTexture(randomTexture, index: 0)
        computeEncoder.setTexture(accumulationTargets[0], index: 1)
        computeEncoder.setTexture(accumulationTargets[1], index: 2)
        computeEncoder.setTexture(splatTargets[0], index: 3)
        computeEncoder.setTexture(splatTargets[1], index: 4)
        computeEncoder.setTexture(finalImage, index: 5)
        computeEncoder.setTexture(scene.environmentMapTexture, index: 6)
        computeEncoder.setBuffer(scene.lightTriangleBuffer, offset: 0, index: 6)
        computeEncoder.setBuffer(scene.instanceLightIndicesBuffer, offset: 0, index: 7)
        computeEncoder.setBuffer(scene.environmentMapCDFBuffer, offset: 0, index: 8)
        computeEncoder.setBuffer(textureArgumentBuffer, offset: 0, index: 9)
        computeEncoder.setBuffer(MaterialRegistry.shared.getBuffer(), offset: 0, index: 10)
                
        let t6 = CFAbsoluteTimeGetCurrent()
        
        let textures = TextureRegistry.shared.getTextures()
        computeEncoder.useResources(textures, usage: .read)
//
//        for geometry in scene.geometries {
//            for resource in geometry.resources() {
//                computeEncoder.useResource(resource, usage: .read)
//            }
//        }
        
        for primitiveAccel in primitiveAccelerationStructures {
            computeEncoder.useResource(primitiveAccel, usage: .read)
        }
//        
        computeEncoder.setComputePipelineState(raytracingPipeline)
        let t7 = CFAbsoluteTimeGetCurrent()
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        let t8 = CFAbsoluteTimeGetCurrent()
        
        computeEncoder.endEncoding()
        let t9 = CFAbsoluteTimeGetCurrent()
        
        // finalizeAtomicBuffer
        guard let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        finalizeEncoder.setComputePipelineState(finalizePipeline)
        finalizeEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 0)
        finalizeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 1)
        finalizeEncoder.setTexture(splatTargets[1], index: 0)
        finalizeEncoder.setTexture(splatTargets[0], index: 1)
        finalizeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        finalizeEncoder.endEncoding()
        
        (accumulationTargets[0], accumulationTargets[1]) = (accumulationTargets[1], accumulationTargets[0])
        (splatTargets[0], splatTargets[1]) = (splatTargets[1], splatTargets[0])
        
        var t10 = CFAbsoluteTimeGetCurrent()
        var t11 = t10
        
        if let currentDrawable = view.currentDrawable {
            t10 = CFAbsoluteTimeGetCurrent()

            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = currentDrawable.texture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
            
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.setRenderPipelineState(copyPipeline)
                renderEncoder.setFragmentTexture(finalImage, index: 0)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                renderEncoder.endEncoding()
            }
            
            commandBuffer.present(currentDrawable)
            t11 = CFAbsoluteTimeGetCurrent()
        }
        
        commandBuffer.commit()
        let t12 = CFAbsoluteTimeGetCurrent()
        
//        print(String(format: """
//            -----------------------------
//            Semaphore wait:     %7.2fms
//            Command buffer:     %7.2fms
//            Camera input:       %7.2fms
//            Uniforms:           %7.2fms
//            Make encoder:       %7.2fms
//            Set buffers/tex:    %7.2fms
//            Set pipeline:       %7.2fms
//            Dispatch:           %7.2fms
//            End encoding:       %7.2fms
//            Get drawable:       %7.2fms
//            Render pass:        %7.2fms
//            Commit:             %7.2fms
//            -----------------------------
//            TOTAL CPU:          %7.2fms
//            """,
//                     (t1 - t0) * 1000,
//                     (t2 - t1) * 1000,
//                     (t3 - t2) * 1000,
//                     (t4 - t3) * 1000,
//                     (t5 - t4) * 1000,
//                     (t6 - t5) * 1000,
//                     (t7 - t6) * 1000,
//                     (t8 - t7) * 1000,
//                     (t9 - t8) * 1000,
//                     (t10 - t9) * 1000,
//                     (t11 - t10) * 1000,
//                     (t12 - t11) * 1000,
//                     (t12 - t0) * 1000))
    }
}

func getManagedBufferStorageMode() -> MTLResourceOptions {
#if os(iOS)
    return []
#else
    return .storageModeShared
#endif
}
