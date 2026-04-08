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
    private var calculateIntersectionsWithCompactionPipeline: MTLComputePipelineState!
    private var sampleBXDFsPipeline: MTLComputePipelineState!
    private var finalizeAccumulationPipeline: MTLComputePipelineState!
    
    private var rayBuffers: [RayBuffers] = []
    private var currentBuffer: Int = 0
    
    private var rayCountBuffer: MTLBuffer!
    private var nextRayCountBuffer: MTLBuffer!
    private var intersectionCountBuffer: MTLBuffer!
    
    private var accumulationBuffer: MTLBuffer!
    
    private var intersectionPositionsBuffer: MTLBuffer!
    private var intersectionNormalsBuffer: MTLBuffer!
    private var intersectionSampledMaterialsBuffer: MTLBuffer!
    private var intersectionLightIndicesBuffer: MTLBuffer!
    private var intersectionEmissionBuffer: MTLBuffer!
        
    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)
        
        createWaveFrontPipelines()
    }
    
    
    private func createWaveFrontPipelines() {
        let createCameraRaysFunction = library.makeFunction(name: "createCameraRays")!
        createCameraRaysPipeline = newComputePipelineState(function: createCameraRaysFunction)
        
        let calculateIntersectionsFunction = library.makeFunction(name: "calculateIntersections")!
        calculateIntersectionsPipeline = newComputePipelineState(function: calculateIntersectionsFunction)
        
        let calculateIntersectionsWithCompactionFunction = library.makeFunction(name: "calculateIntersectionsWithCompaction")!
        calculateIntersectionsWithCompactionPipeline = newComputePipelineState(function: calculateIntersectionsWithCompactionFunction)
        
        let sampleBXDFsFunction = library.makeFunction(name: "sampleBXDFs")!
        sampleBXDFsPipeline = newComputePipelineState(function: sampleBXDFsFunction)
        
        let finalizeAccumulationFunction = library.makeFunction(name: "finalizeAccumulation")!
        finalizeAccumulationPipeline = newComputePipelineState(function: finalizeAccumulationFunction)
    }
    
    private func createWaveFrontBuffers(maxRays: Int) {
        rayBuffers = []
        for _ in 0..<2 {
            rayBuffers.append(RayBuffers(origins: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         directions: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         throughputs: device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)!,
                                         pixelIndices: device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!,
                                         rngStates: device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModePrivate)!,
                                         rayAlive: device.makeBuffer(length: maxRays * MemoryLayout<Bool>.stride, options: .storageModePrivate)!)
            )
        }
        
        rayCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        nextRayCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        intersectionCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        
        accumulationBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
        
        intersectionPositionsBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
        intersectionNormalsBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
        intersectionSampledMaterialsBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SampledMaterial>.stride, options: .storageModePrivate)
        intersectionLightIndicesBuffer = device.makeBuffer(length: maxRays * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intersectionEmissionBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
    }
    
    private func createCameraRays(commandBuffer: MTLCommandBuffer, threadgroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        let initialCount = UInt32(drawableSize.width) * UInt32(drawableSize.height)
        rayCountBuffer.contents().storeBytes(of: initialCount, as: UInt32.self)
        blitEncoder.fill(buffer: nextRayCountBuffer, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
        blitEncoder.endEncoding()
        
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        commandEncoder.setComputePipelineState(createCameraRaysPipeline)
        
        let buffers = [
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngStates,
            rayBuffers[currentBuffer].rayAlive,
            
            rayCountBuffer,
            accumulationBuffer
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        
        commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }
    
    private func calculateIntersections(commandBuffer: MTLCommandBuffer, rayCount: Int) {
        guard rayCount > 0 else { return }
        
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: intersectionCountBuffer, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
        blitEncoder.fill(buffer: rayBuffers[1 - currentBuffer].rayAlive, range: 0..<rayBuffers[1 - currentBuffer].rayAlive.length, value: 0)
        blitEncoder.endEncoding()
        
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        commandEncoder.setComputePipelineState(calculateIntersectionsPipeline)
        
        let nextBuffer = 1 - currentBuffer
        
        let buffers: [MTLBuffer?] = [
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngStates,
            rayBuffers[currentBuffer].rayAlive,
            
            rayCountBuffer,
            
            intersectionCountBuffer,
            intersectionPositionsBuffer,
            intersectionNormalsBuffer,
            intersectionSampledMaterialsBuffer,
            intersectionLightIndicesBuffer,
            intersectionEmissionBuffer,
            
            textureArgumentBuffer,
            MaterialRegistry.shared.getBuffer(),
            scene.instanceLightIndicesBuffer,
            
            instanceBuffer,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        commandEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: buffers.count + 1)
        
        let textures = TextureRegistry.shared.getTextures()
        commandEncoder.useResources(textures, usage: .read)
        for primitiveAccel in primitiveAccelerationStructures {
            commandEncoder.useResource(primitiveAccel, usage: .read)
        }
        
        let threadsPerGroup = calculateIntersectionsPipeline.threadExecutionWidth
        assert(rayCount > 0)
        let threadgroups = (rayCount + threadsPerGroup - 1) / threadsPerGroup
        commandEncoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        commandEncoder.endEncoding()
    }
    
    private func calculateIntersectionsWithCompaction(commandBuffer: MTLCommandBuffer, rayCount: Int) {
        guard rayCount > 0 else { return }
        let nextBuffer = 1 - currentBuffer
        
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: intersectionCountBuffer, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
        blitEncoder.fill(buffer: rayBuffers[nextBuffer].rayAlive, range: 0..<rayBuffers[nextBuffer].rayAlive.length, value: 0)
        blitEncoder.endEncoding()
        
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        commandEncoder.setComputePipelineState(calculateIntersectionsWithCompactionPipeline)
        
        let buffers: [MTLBuffer?] = [
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngStates,
            rayBuffers[currentBuffer].rayAlive,
            
            rayBuffers[nextBuffer].origins,
            rayBuffers[nextBuffer].directions,
            rayBuffers[nextBuffer].throughputs,
            rayBuffers[nextBuffer].pixelIndices,
            rayBuffers[nextBuffer].rngStates,
            rayBuffers[nextBuffer].rayAlive,
            
            rayCountBuffer,
            
            intersectionCountBuffer,
            intersectionPositionsBuffer,
            intersectionNormalsBuffer,
            intersectionSampledMaterialsBuffer,
            intersectionLightIndicesBuffer,
            intersectionEmissionBuffer,
            
            textureArgumentBuffer,
            MaterialRegistry.shared.getBuffer(),
            scene.instanceLightIndicesBuffer,
            
            instanceBuffer,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        commandEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: buffers.count + 1)
        
        let textures = TextureRegistry.shared.getTextures()
        commandEncoder.useResources(textures, usage: .read)
        for primitiveAccel in primitiveAccelerationStructures {
            commandEncoder.useResource(primitiveAccel, usage: .read)
        }
        
        let threadsPerGroup = calculateIntersectionsPipeline.threadExecutionWidth
        assert(rayCount > 0)
        let threadgroups = (rayCount + threadsPerGroup - 1) / threadsPerGroup
        commandEncoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        commandEncoder.endEncoding()
        
        currentBuffer = 1 - currentBuffer
    }
    
    private func sampleBXDFs(commandBuffer: MTLCommandBuffer, rayCount: Int, bounceIndex: UInt32) {
        guard rayCount > 0 else { return }
        
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: nextRayCountBuffer, range: 0..<MemoryLayout<UInt32>.stride, value: 0)
        blitEncoder.endEncoding()
        
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        var bounces = bounceIndex
        let nextBuffer = 1 - currentBuffer
        
        commandEncoder.setComputePipelineState(sampleBXDFsPipeline)
        
        let buffers: [MTLBuffer?] = [
            rayBuffers[currentBuffer].origins,
            rayBuffers[currentBuffer].directions,
            rayBuffers[currentBuffer].throughputs,
            rayBuffers[currentBuffer].pixelIndices,
            rayBuffers[currentBuffer].rngStates,
            
            rayBuffers[nextBuffer].origins,
            rayBuffers[nextBuffer].directions,
            rayBuffers[nextBuffer].throughputs,
            rayBuffers[nextBuffer].pixelIndices,
            rayBuffers[nextBuffer].rngStates,
            
            rayCountBuffer,
            nextRayCountBuffer,
            rayBuffers[currentBuffer].rayAlive,
            accumulationBuffer,
            
            intersectionPositionsBuffer,
            intersectionNormalsBuffer,
            intersectionSampledMaterialsBuffer,
            intersectionLightIndicesBuffer,
            intersectionEmissionBuffer,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            commandEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        commandEncoder.setBytes(&bounces, length: MemoryLayout<UInt32>.stride, index: buffers.count + 1)
        
        let threadsPerGroup = sampleBXDFsPipeline.threadExecutionWidth
        let threadgroups = (rayCount + threadsPerGroup - 1) / threadsPerGroup
        commandEncoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        commandEncoder.endEncoding()
    }
    
    private func finalizeAccumulation(commandBuffer: MTLCommandBuffer, threadgroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        commandEncoder.setComputePipelineState(finalizeAccumulationPipeline)
        
        
        commandEncoder.setBuffer(accumulationBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 1)
        
        commandEncoder.setTexture(finalImage, index: 0)
        
        commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }
    
    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
        
        let requiredPixels = Int(size.width) * Int(size.height)

        if requiredPixels > bufferPixels || requiredPixels < bufferPixels / 2 {
            createWaveFrontBuffers(maxRays: requiredPixels)
            bufferPixels = requiredPixels
        }
    }
    
    override func draw(in view: MTKView) {
        _ = semaphore.wait(timeout: .distantFuture)
        
        guard let commandBuffer = queue.makeCommandBuffer() else {
            return
        }
        
        commandBuffer.addCompletedHandler { _ in
            self.semaphore.signal()
        }
        
        processCameraInput()
        updateUniforms()
        
        var rayCount: Int = Int(drawableSize.width) * Int(drawableSize.height)
        
        let width = Int(drawableSize.width)
        let height = Int(drawableSize.height)
        
        let threadWidth = createCameraRaysPipeline.threadExecutionWidth
        let threadHeight = createCameraRaysPipeline.maxTotalThreadsPerThreadgroup / threadWidth
        
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(width: (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                   height: (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                   depth: 1)
        
        createCameraRays(commandBuffer: commandBuffer, threadgroups: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        for bounce in 0..<11 {
            if rayCount == 0 { break }
            
            guard let bounceCommandBuffer = queue.makeCommandBuffer() else { return }
            
            if false {
                calculateIntersectionsWithCompaction(commandBuffer: bounceCommandBuffer, rayCount: rayCount)
            } else {
                calculateIntersections(commandBuffer: bounceCommandBuffer, rayCount: rayCount)
            }
            
            sampleBXDFs(commandBuffer: bounceCommandBuffer, rayCount: rayCount, bounceIndex: UInt32(bounce))
            
            bounceCommandBuffer.commit()
            bounceCommandBuffer.waitUntilCompleted()
            
            currentBuffer = 1 - currentBuffer
            swap(&rayCountBuffer, &nextRayCountBuffer)
            rayCount = Int(rayCountBuffer.contents().load(as: UInt32.self))
            
//            print("bounce \(bounce): \(rayCount) rays")
        }
        
        guard let finalCommandBuffer = queue.makeCommandBuffer() else { return }
        finalCommandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        finalizeAccumulation(commandBuffer: finalCommandBuffer, threadgroups: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        presentDrawable(view: view, commandBuffer: finalCommandBuffer)
        finalCommandBuffer.commit()
    }
}

struct RayBuffers {
    var origins: MTLBuffer
    var directions: MTLBuffer
    var throughputs: MTLBuffer
    var pixelIndices: MTLBuffer
    var rngStates: MTLBuffer
    var rayAlive: MTLBuffer
}
