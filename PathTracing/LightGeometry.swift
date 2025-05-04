//
//  LightGeometry.swift
//  PathTracing
//
//  Created on 5/2/25.
//

import MetalKit

class LightGeometry: Geometry {
    var lightColors: [SIMD3<Float>] = []
    var lightAmplifier: Float
    
    init(device: MTLDevice, lightAmplifier: Float = 1.0) {
        self.lightAmplifier = lightAmplifier
        super.init(device: device)
    }
        
    override func geometryDescriptor() -> MTLAccelerationStructureGeometryDescriptor? {
        let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        descriptor.vertexBuffer = vertexPositionBuffer
        descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        descriptor.triangleCount = vertices.count / 3
        return descriptor
    }
        
    override func resources() -> [MTLResource] {
        var resourceArray: [MTLResource] = []

        if let nb = vertexNormalBuffer { resourceArray.append(nb) }
        if let cb = vertexColorBuffer { resourceArray.append(cb) }
        if let mb = materialBuffer { resourceArray.append(mb) }
        if let tx = textureCoordinatesBuffer { resourceArray.append(tx) }

        return resourceArray
    }
}
