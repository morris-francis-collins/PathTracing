//
//  MegaKernelRenderer.swift
//  PathTracing
//
//  Created on 2/14/26.
//

import Foundation
import Metal
import MetalKit
import simd
import SwiftUI

class MegaKernelRenderer: Renderer {
    var raytracingPipeline: MTLComputePipelineState!
    var clearBufferPipeline: MTLComputePipelineState!
    var finalizePipeline: MTLComputePipelineState!
    
    var atomicSplatBuffer: MTLBuffer!
    
    var accumulationTargets: [MTLTexture?] = [nil, nil]
    var splatTargets: [MTLTexture?] = [nil, nil]

    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)
        
        createMegaKernelPipelines()
    }
    
    func createMegaKernelPipelines() {
        let raytracingFunction = specializedFunction(named: "raytracingKernel")
        raytracingPipeline = newComputePipelineState(function: raytracingFunction)
                        
        let clearFunction = specializedFunction(named: "clearAtomicBuffer")
        clearBufferPipeline = newComputePipelineState(function: clearFunction)
        
        let finalizeFunction = specializedFunction(named: "finalizeAtomicBuffer")
        finalizePipeline = newComputePipelineState(function: finalizeFunction)
    }
    
    private func createAtomicSplatBuffers(maxRays: Int) {
        let bufferSize = maxRays * 3 * MemoryLayout<Float>.size
        atomicSplatBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        )
    }
    
    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
        let textureDescriptor = getImageTextureDescriptor(size)
        
        for i in 0..<2 {
            accumulationTargets[i] = device.makeTexture(descriptor: textureDescriptor)
        }
        
        for i in 0..<2 {
            splatTargets[i] = device.makeTexture(descriptor: textureDescriptor)
        }
        
        let requiredPixels = Int(size.width) * Int(size.height)

        if requiredPixels > bufferPixels || requiredPixels < bufferPixels / 2 {
            createAtomicSplatBuffers(maxRays: requiredPixels)
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
        
        let width = Int(drawableSize.width)
        let height = Int(drawableSize.height)
        
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: raytracingPipeline)
        
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
        
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
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
        computeEncoder.setBuffer(textureArgumentBuffer, offset: 0, index: 8)
        computeEncoder.setBuffer(MaterialRegistry.shared.getBuffer(), offset: 0, index: 9)
        computeEncoder.setBuffer(scene.lightAliasEntriesBuffer, offset: 0, index: 10)
        computeEncoder.setBuffer(scene.lightTriangleAliasEntriesBuffer, offset: 0, index: 11)
        computeEncoder.setBuffer(scene.environmentMapAliasEntriesBuffer, offset: 0, index: 12)
        
        let textures = TextureRegistry.shared.getTextures()
        computeEncoder.useResources(textures, usage: .read)
        
        for primitiveAccel in primitiveAccelerationStructures {
            computeEncoder.useResource(primitiveAccel, usage: .read)
        }
        
        computeEncoder.setComputePipelineState(raytracingPipeline)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        
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
        
        presentDrawable(view: view, commandBuffer: commandBuffer)
        commandBuffer.commit()
    }
}
