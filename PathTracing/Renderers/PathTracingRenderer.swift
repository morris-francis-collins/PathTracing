//
//  PathTracingRenderer.swift
//  PathTracing
//
//  Created on 4/11/26.
//

import MetalKit

class PathTracingRenderer: Renderer {
    var pathTracingPipeline: MTLComputePipelineState!
    
    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)
        
        createPathTracingPipelines()
    }
    
    func createPathTracingPipelines() {
        let pathTracingFunction = specializedFunction(named: "pathTracingKernel")
        pathTracingPipeline = newComputePipelineState(function: pathTracingFunction)
    }
        
    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
                
        let requiredPixels = Int(size.width) * Int(size.height)
        
        if requiredPixels > bufferPixels || requiredPixels < bufferPixels / 2 {
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
                
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: pathTracingPipeline)
                
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        let buffers = [
            accumulationBuffer,
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
        computeEncoder.setComputePipelineState(pathTracingPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        finalizeAccumulation(commandBuffer: commandBuffer, threadgroups: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        presentDrawable(view: view, commandBuffer: commandBuffer)
        
        commandBuffer.commit()
    }
}
