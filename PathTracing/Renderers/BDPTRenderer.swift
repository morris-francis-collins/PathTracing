//
//  BDPTRenderer.swift
//  PathTracing
//
//  Created on 2/14/26.
//

import Foundation
import Metal
import MetalKit
import simd
import SwiftUI

class BDPTRenderer: Renderer {
    var BDPTPipeline: MTLComputePipelineState!
    
    var atomicSplatBuffer: MTLBuffer!
    
    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)
        
        createBDPTPipelines()
    }
    
    func createBDPTPipelines() {
        let BDPTFunction = specializedFunction(named: "bidirectionalPathTracingKernel")
        BDPTPipeline = newComputePipelineState(function: BDPTFunction)
        
        let finalizeAccumulationFunction = library.makeFunction(name: "finalizeAccumulationBDPT")!
        finalizeAccumulationPipeline = newComputePipelineState(function: finalizeAccumulationFunction)
    }
    
    private func createAtomicSplatBuffers(maxRays: Int) {
        let bufferSize = maxRays * 3 * MemoryLayout<Float>.stride
        atomicSplatBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        )
    }
    
    override func finalizeAccumulation(commandBuffer: MTLCommandBuffer, threadgroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        commandEncoder.setComputePipelineState(finalizeAccumulationPipeline)
        
        commandEncoder.setBuffer(accumulationBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 2)
        commandEncoder.setTexture(finalImage, index: 0)
        
        commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }

    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
        
        let requiredPixels = Int(size.width) * Int(size.height)
        createAtomicSplatBuffers(maxRays: requiredPixels)
        bufferPixels = requiredPixels
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
                
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: BDPTPipeline)
                
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        let buffers = [
            accumulationBuffer,
            atomicSplatBuffer,
            scene.lightBuffer,
            scene.lightTriangleBuffer,
            scene.instanceLightIndicesBuffer,
            scene.lightAliasEntriesBuffer,
            scene.lightTriangleAliasEntriesBuffer,
            scene.environmentMapAliasEntriesBuffer,
            
            MaterialRegistry.shared.getBuffer(),
            textureArgumentBuffer,
        ]
        
        let textures = [
            randomTexture,
            scene.environmentMapTexture,
        ]
        
        for (i, buffer) in buffers.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        for (i, texture) in textures.enumerated() {
            computeEncoder.setTexture(texture, index: i)
        }

        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
        computeEncoder.setBuffer(instanceBuffer, offset: 0, index: buffers.count + 1)
        computeEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: buffers.count + 2)

        // bind resources
        computeEncoder.useResources(TextureRegistry.shared.getTextures(), usage: .read)
        for primitiveAccel in primitiveAccelerationStructures { computeEncoder.useResource(primitiveAccel, usage: .read) }
        
        // dispatch and display
        computeEncoder.setComputePipelineState(BDPTPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
                
        finalizeAccumulation(commandBuffer: commandBuffer, threadgroups: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        presentDrawable(view: view, commandBuffer: commandBuffer)
        
        commandBuffer.commit()
    }
}
