//
//  WaveFrontRenderer.swift
//  PathTracing
//
//  Created on 2/14/26.
//

import Foundation
import Metal
import MetalKit
import simd
import SwiftUI

class WaveFrontRenderer: Renderer {
    private var createCameraRaysPipeline: MTLComputePipelineState!
    private var calculateIntersectionsPipeline: MTLComputePipelineState!
    private var handleEscapedRaysPipeline: MTLComputePipelineState!
    private var sampleBXDFsPipeline: MTLComputePipelineState!
    
    private var rayBuffers: [RayBuffers] = []
    private var currentBuffer: Int = 0
        
    private var escapedQueueBuffer: MTLBuffer!
    private var intersectedQueueBuffer: MTLBuffer!
    
    private var intersectionResultBuffer: MTLBuffer!
    
    private var rayCountBuffer: MTLBuffer!
    private var rayCount: UInt32 { rayCountBuffer.contents().load(as: UInt32.self) }
    
    private var escapedRayCountBuffer: MTLBuffer!
    private var intersectedRayCountBuffer: MTLBuffer!
    private var survivedRayCountBuffer: MTLBuffer!
    
    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)
        createWaveFrontPipelines()
    }
    
    private func createWaveFrontPipelines() {
        createCameraRaysPipeline = newComputePipelineState(function: library.makeFunction(name: "createCameraRays")!)
        calculateIntersectionsPipeline = newComputePipelineState(function: library.makeFunction(name: "calculateIntersections")!)
        handleEscapedRaysPipeline = newComputePipelineState(function: library.makeFunction(name: "handleEscapedRays")!)
        sampleBXDFsPipeline = newComputePipelineState(function: library.makeFunction(name: "sampleBXDFs")!)
    }
    
    private func createWaveFrontBuffers(maxRays: Int) {
        rayBuffers = []
        for _ in 0..<2 {
            rayBuffers.append(RayBuffers(origins: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         directions: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         throughputs: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         pixelIndices: device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!,
                                         rngDimensions: device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!)
            )
        }
        
        accumulationBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
                
        escapedQueueBuffer = device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)
        intersectedQueueBuffer = device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)
        
        intersectionResultBuffer = device.makeBuffer(length: maxRays * Int(INTERSECTION_RESULT_STRIDE), options: .storageModePrivate)
        
        var initialRays = maxRays
        rayCountBuffer = device.makeBuffer(bytes: &initialRays, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        escapedRayCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        intersectedRayCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        survivedRayCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    }
    
    private func createCameraRays(_ commandEncoder: MTLComputeCommandEncoder) {
        commandEncoder.setComputePipelineState(createCameraRaysPipeline)
        
        let buffers = [
            accumulationBuffer,
            
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngDimensions,
            
            scene.sobolData.buffer,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        
        // dispatch
        let (threadgroups, threadsPerThreadgroup) = getDispatchSize2D(pipeline: createCameraRaysPipeline)
        commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    
    private func calculateIntersections(_ commandEncoder: MTLComputeCommandEncoder) {
        commandEncoder.setComputePipelineState(calculateIntersectionsPipeline)
        
        let buffers = [
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            
            intersectionResultBuffer,
            
            escapedQueueBuffer,
            intersectedQueueBuffer,
            
            rayCountBuffer,
            escapedRayCountBuffer,
            intersectedRayCountBuffer,
            
            textureArgumentBuffer,
            MaterialRegistry.shared.getBuffer(),
            scene.instanceLightIndicesBuffer,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }

        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        commandEncoder.setBuffer(instanceBuffer, offset: 0, index: buffers.count + 1)
        commandEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: buffers.count + 2)
        
        // bind resources
        commandEncoder.useResources(TextureRegistry.shared.getTextures(), usage: .read)
        for primitiveAccel in primitiveAccelerationStructures { commandEncoder.useResource(primitiveAccel, usage: .read) }
        
        // dispatch
        let threadsPerGroup = calculateIntersectionsPipeline.threadExecutionWidth
        let threadgroups = (Int(rayCount) + threadsPerGroup - 1) / threadsPerGroup
        
        commandEncoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    }
    
    private func handleEscapedRays(_ commandEncoder: MTLComputeCommandEncoder) {
        commandEncoder.setComputePipelineState(handleEscapedRaysPipeline)
        
        let buffers = [
            accumulationBuffer,
            
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            
            escapedQueueBuffer,
            
            escapedRayCountBuffer,
        ]
        
        let textures = [
            scene.environmentMapTexture
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        for (i, texture) in textures.enumerated() {
            commandEncoder.setTexture(texture, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        
        // bind resources
        commandEncoder.useResources(TextureRegistry.shared.getTextures(), usage: .read) // TODO: need ts?
        for primitiveAccel in primitiveAccelerationStructures { commandEncoder.useResource(primitiveAccel, usage: .read) }
        
        // dispatch
        let threadsPerGroup = handleEscapedRaysPipeline.threadExecutionWidth
        let threadgroups = (Int(rayCount) + threadsPerGroup - 1) / threadsPerGroup
        
        commandEncoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    }
    
    
    private func sampleBXDFs(_ commandEncoder: MTLComputeCommandEncoder, bounceIndex: UInt32) {
        var bounces = bounceIndex
        let nextBuffer = 1 - currentBuffer
        
        commandEncoder.setComputePipelineState(sampleBXDFsPipeline)
        
        let buffers: [MTLBuffer?] = [
            accumulationBuffer,
            
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngDimensions,
            
            rayBuffers[nextBuffer].origins,
            rayBuffers[nextBuffer].directions,
            rayBuffers[nextBuffer].throughputs,
            rayBuffers[nextBuffer].pixelIndices,
            rayBuffers[nextBuffer].rngDimensions,
            
            intersectionResultBuffer,
            
            intersectedQueueBuffer,
            
            intersectedRayCountBuffer,
            survivedRayCountBuffer,

            scene.sobolData.buffer,
            textureArgumentBuffer,
            MaterialRegistry.shared.getBuffer(),
            scene.instanceLightIndicesBuffer
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBytes(&bounces, length: MemoryLayout<UInt32>.stride, index: buffers.count)
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        commandEncoder.setBuffer(instanceBuffer, offset: 0, index: buffers.count + 1)
        commandEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: buffers.count + 2)
        commandEncoder.setBytes(&bounces, length: MemoryLayout<UInt32>.stride, index: buffers.count + 3)

        // bind resources
        commandEncoder.useResources(TextureRegistry.shared.getTextures(), usage: .read) // TODO: need ts?
        for primitiveAccel in primitiveAccelerationStructures { commandEncoder.useResource(primitiveAccel, usage: .read) }
        
        // dispatch
        let threadsPerGroup = sampleBXDFsPipeline.threadExecutionWidth
        let threadgroups = (Int(rayCount) + threadsPerGroup - 1) / threadsPerGroup
        commandEncoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    }
    
    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
        
        let requiredPixels = Int(size.width) * Int(size.height)
        
        createWaveFrontBuffers(maxRays: requiredPixels)
        bufferPixels = requiredPixels
    }
    
    override func draw(in view: MTKView) {
        _ = semaphore.wait(timeout: .distantFuture)
                
        processCameraInput()
        updateUniforms()
        
        // ============ Camera rays ============
        guard let setupCmdBuffer = queue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }
                
        setUInt32BufferValue(rayCountBuffer, value: Int(drawableSize.width) * Int(drawableSize.height))
        
        let setupEncoder = setupCmdBuffer.makeComputeCommandEncoder()!
        createCameraRays(setupEncoder)
        setupEncoder.endEncoding()
        
        setupCmdBuffer.commit()
        setupCmdBuffer.waitUntilCompleted()
                
        for bounce in 0..<MAX_PATH_LENGTH {
            if rayCount == 0 { break }
            
            setUInt32BufferValue(escapedRayCountBuffer, value: 0)
            setUInt32BufferValue(intersectedRayCountBuffer, value: 0)
            setUInt32BufferValue(survivedRayCountBuffer, value: 0)

            guard let bounceCmdBuffer = queue.makeCommandBuffer() else { break }
            let encoder = bounceCmdBuffer.makeComputeCommandEncoder()!
            
            calculateIntersections(encoder)
            encoder.memoryBarrier(scope: .buffers)
            
            handleEscapedRays(encoder)
            encoder.memoryBarrier(scope: .buffers)
            
            sampleBXDFs(encoder, bounceIndex: UInt32(bounce))
            
            encoder.endEncoding()
            bounceCmdBuffer.commit()
            bounceCmdBuffer.waitUntilCompleted()
            
            // Swap for next bounce
            currentBuffer = 1 - currentBuffer
            swap(&rayCountBuffer, &survivedRayCountBuffer)
            print("bounce \(bounce): \(rayCount) rays")
        }
        
        // ============ Finalize + present ============
        guard let finalCmdBuffer = queue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }
        finalCmdBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        
        finalizeAccumulation(commandBuffer: finalCmdBuffer)
        presentDrawable(view: view, commandBuffer: finalCmdBuffer)
        
        finalCmdBuffer.commit()
        
        print("------")
    }
}

func setUInt32BufferValue(_ buffer: MTLBuffer, value: Int) {
    buffer.contents().storeBytes(of: UInt32(value), as: UInt32.self)
}
    
struct RayBuffers {
    var origins: MTLBuffer
    var directions: MTLBuffer
    var throughputs: MTLBuffer
    var pixelIndices: MTLBuffer
    var rngDimensions: MTLBuffer
}
