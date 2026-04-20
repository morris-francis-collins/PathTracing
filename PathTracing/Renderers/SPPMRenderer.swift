//
//  SPPMRenderer.swift
//  PathTracing
//
//  Created on 4/13/26.
//

import MetalKit

class SPPMRenderer: Renderer {
    private let initialRadius: Float = 0.001 // TODO: adapt with scene size
    
    private var createCameraRaysPipeline: MTLComputePipelineState!
    private var generateHitPointsPipeline: MTLComputePipelineState!
    private var createHashGridPipeline: MTLComputePipelineState!
    private var tracePhotonsPipeline: MTLComputePipelineState!
        
    private var rayOriginBuffer: MTLBuffer!
    private var rayDirectionBuffer: MTLBuffer!
    
    private var hitPointBSDFBuffer: MTLBuffer!
    private var hitPointLocationBuffer: MTLBuffer!
    private var hitPointIncomingDirectionBuffer: MTLBuffer!
    private var hitPointNormalBuffer: MTLBuffer!
    private var hitPointHashBuffer: MTLBuffer!
    
    private var hashTableBSDFBuffer: MTLBuffer!
    private var hashTableLocationBuffer: MTLBuffer!
    private var hashTableIncomingDirectionBuffer: MTLBuffer!
    private var hashTableNormalBuffer: MTLBuffer!
    private var hashTableShadingPixelBuffer: MTLBuffer!

    private var hashTableCountBuffer: MTLBuffer!
    private var hashTableOffsetBuffer: MTLBuffer!
    private var hashTableIndexBuffer: MTLBuffer!
    
    private var totalPhotonCountBuffer: MTLBuffer!
    private var currentPhotonCountBuffer: MTLBuffer!
    private var gatheringRadiusBuffer: MTLBuffer!
    
    private var hashGridSizeBuffer: MTLBuffer!
    private var newHashGridSizeBuffer: MTLBuffer!
    
    private var SPPMAccumulationBuffer: MTLBuffer!

    override init(device: any MTLDevice, scene: GameScene) {
        super.init(device: device, scene: scene)

        createSPPMPipelines()
    }
    
    
    private func createSPPMPipelines() {
        createCameraRaysPipeline = newComputePipelineState(function: library.makeFunction(name: "createCameraRaysSPPM")!)
        generateHitPointsPipeline = newComputePipelineState(function: library.makeFunction(name: "generateHitPointsSPPM")!)
        createHashGridPipeline = newComputePipelineState(function: library.makeFunction(name: "createHashGridSPPM")!)
        tracePhotonsPipeline = newComputePipelineState(function: library.makeFunction(name: "tracePhotonsSPPM")!)
        
        finalizeAccumulationPipeline = newComputePipelineState(function: library.makeFunction(name: "finalizeAccumulationSPPM")!)
    }
    
    private func createSPPMBuffers(maxRays: Int) {
        rayOriginBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        rayDirectionBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        
        hitPointBSDFBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hitPointLocationBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hitPointIncomingDirectionBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hitPointNormalBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hitPointHashBuffer = device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        
        hashTableBSDFBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hashTableLocationBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hashTableIncomingDirectionBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hashTableNormalBuffer = device.makeBuffer(length: maxRays * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        hashTableShadingPixelBuffer = device.makeBuffer(length: Int(HASH_TABLE_SIZE) * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)
        
        hashTableCountBuffer = device.makeBuffer(length: Int(HASH_TABLE_SIZE) * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        hashTableOffsetBuffer = device.makeBuffer(length: Int(HASH_TABLE_SIZE) * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        hashTableIndexBuffer = device.makeBuffer(length: Int(HASH_TABLE_SIZE) * MemoryLayout<UInt32>.stride, options: .storageModeShared) // shared 0 initalizes
        
        totalPhotonCountBuffer = device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        currentPhotonCountBuffer = device.makeBuffer(length: maxRays * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        gatheringRadiusBuffer = device.makeBuffer(length: maxRays * MemoryLayout<Float>.stride, options: .storageModeShared)
        let radPtr = gatheringRadiusBuffer!.contents().bindMemory(to: Float.self, capacity: maxRays)
        for i in 0..<maxRays { radPtr[i] = initialRadius }
        
        hashGridSizeBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        newHashGridSizeBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        hashGridSizeBuffer!.contents().bindMemory(to: Float.self, capacity: 1).pointee = initialRadius
        newHashGridSizeBuffer!.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0
        
        SPPMAccumulationBuffer = device.makeBuffer(length: maxRays * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
    }
    
    private func createCameraRays(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(createCameraRaysPipeline)

        let buffers = [
            rayOriginBuffer,
            rayDirectionBuffer,
            
            scene.sobolData.buffer
        ]
        
        for (i, buffer) in buffers.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: i)
        }

        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)

        // dispatch
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: createCameraRaysPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func generateHitPoints(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(generateHitPointsPipeline)

        let buffers = [
            SPPMAccumulationBuffer,
            
            rayOriginBuffer,
            rayDirectionBuffer,
            
            scene.lightBuffer,
            scene.lightTriangleBuffer,
            scene.instanceLightIndicesBuffer,
            scene.lightAliasEntriesBuffer,
            scene.lightTriangleAliasEntriesBuffer,
            scene.environmentMapAliasEntriesBuffer,
            
            MaterialRegistry.shared.getBuffer(),
            textureArgumentBuffer,

            hitPointBSDFBuffer,
            hitPointLocationBuffer,
            hitPointIncomingDirectionBuffer,
            hitPointNormalBuffer,
            hitPointHashBuffer,
            
            hashTableCountBuffer,
            hashGridSizeBuffer,
            
            scene.sobolData.buffer
        ]
        
        let textures = [
            scene.environmentMapTexture
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

        // dispatch
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: generateHitPointsPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func createHashGrid(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(createHashGridPipeline)

        let buffers = [
            hitPointBSDFBuffer,
            hitPointLocationBuffer,
            hitPointIncomingDirectionBuffer,
            hitPointNormalBuffer,
            hitPointHashBuffer,

            hashTableBSDFBuffer,
            hashTableLocationBuffer,
            hashTableIncomingDirectionBuffer,
            hashTableNormalBuffer,
            hashTableShadingPixelBuffer,
            
            totalPhotonCountBuffer,
            currentPhotonCountBuffer,
            gatheringRadiusBuffer,

            hashTableOffsetBuffer,
            hashTableIndexBuffer,
            newHashGridSizeBuffer
        ]
        
        for (i, buffer) in buffers.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: i)
        }
        
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: buffers.count)
                
        // dispatch
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: createHashGridPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func tracePhotons(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(tracePhotonsPipeline)

        let buffers = [
            SPPMAccumulationBuffer,
            
            scene.lightBuffer,
            scene.lightTriangleBuffer,
            scene.instanceLightIndicesBuffer,
            scene.lightAliasEntriesBuffer,
            scene.lightTriangleAliasEntriesBuffer,
            scene.environmentMapAliasEntriesBuffer,
            
            MaterialRegistry.shared.getBuffer(),
            textureArgumentBuffer,

            hashTableBSDFBuffer,
            hashTableLocationBuffer,
            hashTableIncomingDirectionBuffer,
            hashTableNormalBuffer,
            hashTableShadingPixelBuffer,
            
            hashTableOffsetBuffer,
            hashTableCountBuffer,
            hashGridSizeBuffer,
            
            currentPhotonCountBuffer,
            gatheringRadiusBuffer,
            
            scene.sobolData.buffer
        ]
        
        let textures = [
            scene.environmentMapTexture
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

        // dispatch
        let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: Int(PHOTON_COUNT), height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
    }
    
    override func finalizeAccumulation(commandBuffer: MTLCommandBuffer) {
        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        commandEncoder.setComputePipelineState(finalizeAccumulationPipeline)
        
        commandEncoder.setBuffer(SPPMAccumulationBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 1)
        commandEncoder.setTexture(finalImage, index: 0)
        
        let (threadsPerThreadgroup, threadgroups) = getDispatchSize2D(pipeline: finalizeAccumulationPipeline)
        commandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
    }
    
    func computePrefixSum() {
        let countPtr = hashTableCountBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(HASH_TABLE_SIZE))
        let offsetPtr = hashTableOffsetBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(HASH_TABLE_SIZE))
        var sum: UInt32 = 0
        for i in 0..<HASH_TABLE_SIZE {
            offsetPtr[Int(i)] = sum
            sum += countPtr[Int(i)]
        }
    }

    override func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        super.mtkView(view, drawableSizeWillChange: size)
        
        let requiredPixels = Int(size.width) * Int(size.height)

        createSPPMBuffers(maxRays: requiredPixels)
        bufferPixels = requiredPixels
    }
    
    override func draw(in view: MTKView) {
        _ = semaphore.wait(timeout: .distantFuture)
    
        guard let commandBuffer = queue.makeCommandBuffer() else {
            return
        }
        
        processCameraInput()
        updateUniforms()
        
        createCameraRays(commandBuffer: commandBuffer)
        
        let blit1 = commandBuffer.makeBlitCommandEncoder()!
        blit1.fill(buffer: hashTableCountBuffer!, range: 0..<hashTableCountBuffer!.length, value: 0)
        blit1.endEncoding()
        
        generateHitPoints(commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        computePrefixSum()
        
        guard let commandBuffer222 = queue.makeCommandBuffer() else {
            return
        }
        
        commandBuffer222.addCompletedHandler { _ in
            self.semaphore.signal()
        }
                
        let blit2 = commandBuffer222.makeBlitCommandEncoder()!
        blit2.fill(buffer: hashTableIndexBuffer!, range: 0..<hashTableIndexBuffer!.length, value: 0)
        blit2.endEncoding()

        createHashGrid(commandBuffer: commandBuffer222)
        tracePhotons(commandBuffer: commandBuffer222)
        
        finalizeAccumulation(commandBuffer: commandBuffer222)
        presentDrawable(view: view, commandBuffer: commandBuffer222)
        
        commandBuffer222.commit()
        commandBuffer222.waitUntilCompleted()
        
        // update radius
        let newGridBits = newHashGridSizeBuffer!.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        let newGridSize = Float(bitPattern: newGridBits)
        if newGridSize > 1e-8 {
            hashGridSizeBuffer!.contents().bindMemory(to: Float.self, capacity: 1).pointee = newGridSize
        }
        newHashGridSizeBuffer!.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        print("hashGridSize:", hashGridSizeBuffer!.contents().bindMemory(to: Float.self, capacity: 1).pointee)
    }
}
