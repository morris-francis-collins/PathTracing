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
        
        createWaveFrontPipelines()
        createWaveFrontBuffers()
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
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
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
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
    
    func drawMegaKernel(view: MTKView) {
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
        
        let threadWidth = raytracingPipeline.threadExecutionWidth
        let threadHeight = raytracingPipeline.maxTotalThreadsPerThreadgroup / threadWidth
//        let threadWidth = 16
//        let threadHeight = 16
        
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
        computeEncoder.setBuffer(scene.environmentMapCDFBuffer, offset: 0, index: 8)
        computeEncoder.setBuffer(textureArgumentBuffer, offset: 0, index: 9)
        computeEncoder.setBuffer(MaterialRegistry.shared.getBuffer(), offset: 0, index: 10)
        
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
        
        if let currentDrawable = view.currentDrawable {
            
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
        }
        
        commandBuffer.commit()
    }
    
    struct RayBuffers {
        var origins: MTLBuffer
        var directions: MTLBuffer
        var throughputs: MTLBuffer
        var pixelIndices: MTLBuffer
        var rngStates: MTLBuffer
        var rayAlive: MTLBuffer
    }
    
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
    
    
    private func createWaveFrontBuffers() {
        let maxRays = Int(4 * PIXEL_WIDTH * PIXEL_HEIGHT)
        
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
    
    private func drawWavefront(view: MTKView) {
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
        
        let threadWidth = raytracingPipeline.threadExecutionWidth
        let threadHeight = raytracingPipeline.maxTotalThreadsPerThreadgroup / threadWidth
        
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
            
            print("bounce \(bounce): \(rayCount) rays")
        }
        
        guard let finalCommandBuffer = queue.makeCommandBuffer() else { return }
        finalCommandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        finalizeAccumulation(commandBuffer: finalCommandBuffer, threadgroups: threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        if let currentDrawable = view.currentDrawable {
            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = currentDrawable.texture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
            
            if let renderEncoder = finalCommandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.setRenderPipelineState(copyPipeline)
                renderEncoder.setFragmentTexture(finalImage, index: 0)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                renderEncoder.endEncoding()
            }
            
            finalCommandBuffer.present(currentDrawable)
            finalCommandBuffer.commit()
        }
    }
    
    func draw(in view: MTKView) {
        drawWavefront(view: view)
        //        drawMegaKernel(view: view)
    }
}

func getManagedBufferStorageMode() -> MTLResourceOptions {
#if os(iOS)
    return []
#else
    return .storageModeShared
#endif
}
