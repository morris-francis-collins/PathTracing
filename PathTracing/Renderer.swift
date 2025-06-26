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
    
    var resourcesStride: Int = 0
    var useIntersectionFunctions: Bool = false
    
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
        
    func newComputePipelineState(function: MTLFunction, linkedFunctions: [MTLFunction]?) -> MTLComputePipelineState {
        let linkedFuncs: MTLLinkedFunctions?
        if let linkedFunctions = linkedFunctions {
            linkedFuncs = MTLLinkedFunctions()
            linkedFuncs?.functions = linkedFunctions
        } else {
            linkedFuncs = nil
        }
        
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.linkedFunctions = linkedFuncs
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        
        do {
            let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            return pipeline
        } catch {
            fatalError("Failed to create compute pipeline state for function \(function.name): \(error)")
        }
    }
    
    func specializedFunction(named name: String) -> MTLFunction {
        let constants = MTLFunctionConstantValues()
        var resourcesStride32 = UInt32(resourcesStride)
        constants.setConstantValue(&resourcesStride32, type: .uint, index: 0)
        var useIntersection = useIntersectionFunctions
        constants.setConstantValue(&useIntersection, type: .bool, index: 1)
        
        do {
            let function = try library.makeFunction(name: name, constantValues: constants)
            return function
        } catch {
            fatalError("Failed to create specialized function \(name): \(error)")
        }
    }
    
    func createPipelines() {
        useIntersectionFunctions = false
        for geometry in scene.geometries {
            if geometry.intersectionFunctionName() != nil {
                useIntersectionFunctions = true
                break
            }
        }
        
        var intersectionFunctions: [String: MTLFunction] = [:]
        for geometry in scene.geometries {
            guard let intersectionName = geometry.intersectionFunctionName(),
                  intersectionFunctions[intersectionName] == nil else {
                continue
            }
            let intersectionFunction = specializedFunction(named: intersectionName)
            intersectionFunctions[intersectionName] = intersectionFunction
        }
        
        let raytracingFunction = specializedFunction(named: "raytracingKernel")
        raytracingPipeline = newComputePipelineState(function: raytracingFunction, linkedFunctions: Array(intersectionFunctions.values))
        
        if useIntersectionFunctions {
            let descriptor = MTLVisibleFunctionTableDescriptor()
            descriptor.functionCount = scene.geometries.count
            visibleFunctionTable = raytracingPipeline.makeVisibleFunctionTable(descriptor: descriptor)
            
            for (index, geometry) in scene.geometries.enumerated() {
                if let intersectionName = geometry.intersectionFunctionName(),
                   let intersectionFunction = intersectionFunctions[intersectionName],
                   let table = visibleFunctionTable {
                    let handle = raytracingPipeline.functionHandle(function: intersectionFunction)
                    table.setFunction(handle, index: index)
                }
            }
        }
        
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        do {
            copyPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        } catch {
            fatalError("Failed to create render pipeline state: \(error)")
        }
        
        let clearFunction = library.makeFunction(name: "clearAtomicBuffer")
        clearBufferPipeline = try! device.makeComputePipelineState(function: clearFunction!)
        
        let finalizeFunction = library.makeFunction(name: "finalizeAtomicBuffer")
        finalizePipeline = try! device.makeComputePipelineState(function: finalizeFunction!)
    }
    
    func newArgumentEncoder(forResources resources: [MTLResource]) -> MTLArgumentEncoder {
        var arguments: [MTLArgumentDescriptor] = []
        
        for (index, resource) in resources.enumerated() {
            let argDesc = MTLArgumentDescriptor()
            argDesc.index = index
            argDesc.access = .readOnly
            if resource is MTLBuffer {
                argDesc.dataType = .pointer
            } else if let texture = resource as? MTLTexture {
                argDesc.dataType = .texture
                argDesc.textureType = texture.textureType
            }
            arguments.append(argDesc)
        }
        
        guard let encoder = device.makeArgumentEncoder(arguments: arguments) else {
            fatalError("Failed to create argument encoder.")
        }
        
        return encoder
    }
    
    func createBuffers() {
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight
        let options: MTLResourceOptions = getManagedBufferStorageMode()
        uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: options)
        
        scene.uploadToBuffers()
        
        resourcesStride = 0
        for geometry in scene.geometries {
            let encoder = newArgumentEncoder(forResources: geometry.resources())
            resourcesStride = max(resourcesStride, encoder.encodedLength)
        }
        
        resourceBuffer = device.makeBuffer(length: resourcesStride * scene.geometries.count, options: options)
        
        for (geometryIndex, geometry) in scene.geometries.enumerated() {
            let encoder = newArgumentEncoder(forResources: geometry.resources())
            encoder.setArgumentBuffer(resourceBuffer, offset: resourcesStride * geometryIndex)
            for (argumentIndex, resource) in geometry.resources().enumerated() {
                if let bufferResource = resource as? MTLBuffer {
                    encoder.setBuffer(bufferResource, offset: 0, index: argumentIndex)
                } else if let textureResource = resource as? MTLTexture {
                    encoder.setTexture(textureResource, index: argumentIndex)
                }
            }
        }
        
        let bufferSize = Int(2 * 800) * Int(2 * 600) * 3 * MemoryLayout<Float>.size
        atomicSplatBuffer = device.makeBuffer(length: bufferSize,
                                              options: .storageModeManaged)
        
        #if !os(iOS)
        resourceBuffer.didModifyRange(0..<resourceBuffer.length)
        atomicSplatBuffer.didModifyRange(0..<atomicSplatBuffer.length)
        #endif
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
                accelDescriptor.usage = .extendedLimits
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
            let transformMatrix = instance.getPackedTransform()
            instanceDescriptors[instanceIndex].transformationMatrix = transformMatrix
        }
        
        #if !os(iOS)
        instanceBuffer.didModifyRange(0..<instanceBuffer.length)
        #endif
        
        let accelDescriptor = MTLInstanceAccelerationStructureDescriptor()
        accelDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
        accelDescriptor.instanceCount = instanceDescriptorCount
        accelDescriptor.instanceDescriptorBuffer = instanceBuffer
        accelDescriptor.usage = .extendedLimits
        
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
        textureDescriptor.storageMode = .managed // change????
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
        
        #if !os(iOS)
        uniformBuffer.didModifyRange(uniformBufferOffset..<uniformBufferOffset + alignedUniformsSize)
        #endif
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
    
    func draw(in view: MTKView) {
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
        
        guard let clearEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        clearEncoder.setComputePipelineState(clearBufferPipeline)
        clearEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 0)
        clearEncoder.setTexture(splatTargets[1], index: 0)

        
        let clearThreadsPerGroup = MTLSize(width: 32, height: 32, depth: 1)
        let clearThreadgroups = MTLSize(width: (width + clearThreadsPerGroup.width - 1) / clearThreadsPerGroup.width,
                                        height: (height + clearThreadsPerGroup.height - 1) / clearThreadsPerGroup.height,
                                        depth: 1)
        clearEncoder.dispatchThreadgroups(clearThreadgroups, threadsPerThreadgroup: clearThreadsPerGroup)
        clearEncoder.endEncoding()
        
        let threadsPerThreadgroup = MTLSize(width: 32, height: 32, depth: 1)
        let threadgroups = MTLSize(width: (width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                   height: (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                   depth: 1)
                
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
        computeEncoder.setBuffer(resourceBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(instanceBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(scene.lightBuffer, offset: 0, index: 3)
        computeEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: 4)
        computeEncoder.setVisibleFunctionTable(visibleFunctionTable, bufferIndex: 5)
        computeEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 6)
        computeEncoder.setTexture(randomTexture, index: 0)
        computeEncoder.setTexture(accumulationTargets[0], index: 1)
        computeEncoder.setTexture(accumulationTargets[1], index: 2)
        computeEncoder.setTexture(splatTargets[0], index: 3)
        computeEncoder.setTexture(splatTargets[1], index: 4)
        computeEncoder.setTexture(finalImage, index: 5)
        computeEncoder.setBuffer(scene.lightTriangleBuffer, offset: 0, index: 7)
        
        let allTextures = TextureRegistry.shared.getTextures()
        for (index, texture) in allTextures.enumerated() {
            computeEncoder.setTexture(texture, index: 8 + index)
        }

        for geometry in scene.geometries {
            for resource in geometry.resources() {
                computeEncoder.useResource(resource, usage: .read)
            }
        }
        
        for primitiveAccel in primitiveAccelerationStructures {
            computeEncoder.useResource(primitiveAccel, usage: .read)
        }
        
        computeEncoder.setComputePipelineState(raytracingPipeline)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        guard let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        finalizeEncoder.setComputePipelineState(finalizePipeline)
        finalizeEncoder.setBuffer(atomicSplatBuffer, offset: 0, index: 0)
        finalizeEncoder.setTexture(splatTargets[1], index: 0)
        finalizeEncoder.setTexture(splatTargets[0], index: 1)
        finalizeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 2)
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
}

func getManagedBufferStorageMode() -> MTLResourceOptions {
    #if os(iOS)
    return []
    #else
    return .storageModeManaged
    #endif
}
