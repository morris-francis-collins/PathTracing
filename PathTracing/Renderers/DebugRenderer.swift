//
//  DebugRenderer.swift
//  PathTracing
//
//  Created on 3/26/26.
//

import MetalKit

class DebugRenderer: Renderer {
    private var debugType: DebugType!
    private var debugSurfacePropertiesPipeline: MTLComputePipelineState!
    
    init(device: any MTLDevice, scene: GameScene, debugType: DebugType) {
        self.debugType = debugType
        super.init(device: device, scene: scene)
        
        createDebugRendererPipelines()
    }
    
    private func createDebugRendererPipelines() {
        let debugSurfacePropertiesFunction = specializedFunction(named: "debugSurfaceProperties")
        debugSurfacePropertiesPipeline = newComputePipelineState(function: debugSurfacePropertiesFunction)
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
                
        let threadWidth = debugSurfacePropertiesPipeline.threadExecutionWidth
        let threadHeight = debugSurfacePropertiesPipeline.maxTotalThreadsPerThreadgroup / threadWidth
        
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(width: (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                   height: (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                   depth: 1)

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        computeEncoder.setBuffer(scene.instanceLightIndicesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(textureArgumentBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(MaterialRegistry.shared.getBuffer(), offset: 0, index: 2)
        
        computeEncoder.setBuffer(instanceBuffer, offset: 0, index: 3)
        computeEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: 4)
        
        computeEncoder.setBuffer(uniformBuffer, offset: 0, index: 5)
        computeEncoder.setBytes(&debugType, length: MemoryLayout<DebugType>.stride, index: 6)
        
        computeEncoder.setTexture(finalImage, index: 0)
        computeEncoder.setTexture(scene.environmentMapTexture, index: 1)
        
        let textures = TextureRegistry.shared.getTextures()
        computeEncoder.useResources(textures, usage: .read)
        
        for primitiveAccel in primitiveAccelerationStructures {
            computeEncoder.useResource(primitiveAccel, usage: .read)
        }
        
        computeEncoder.setComputePipelineState(debugSurfacePropertiesPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        presentDrawable(view: view, commandBuffer: commandBuffer)
        commandBuffer.commit()
    }
}
