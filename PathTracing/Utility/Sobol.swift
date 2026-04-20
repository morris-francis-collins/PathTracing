//
//  Sobol.swift
//  PathTracing
//
//  Created on 4/19/26.
//

import MetalKit

class Sobol {
    let buffer: MTLBuffer!
    let dimensions: Int
    
    init(device: MTLDevice, dimensions: Int) {
        self.dimensions = dimensions
        
        let length = dimensions * 32
        var data = [UInt32](repeating: 0, count: length)
        fillSobolBuffer(&data, Int32(dimensions))
        
        self.buffer = device.makeBuffer(bytes: data,
                                   length: length * MemoryLayout<UInt32>.stride,
                                   options: .storageModeShared)!
    }
}
