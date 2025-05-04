//
//  SphereGeometry.swift
//  PathTracing
//
//  Created on 4/5/25.
//

import MetalKit

class SphereGeometry: Geometry {
    private var spheres: [Sphere] = []
    
    private var boundingBoxBuffer: MTLBuffer?
    var sphereBuffer: MTLBuffer?
    var reflectanceBuffer: MTLBuffer?
    var refractionBuffer: MTLBuffer?
    var opacityBuffer: MTLBuffer?

    var reflectance: Float = 0.0
    var refraction: Float = 1.0
    var opacity: Float = 1.0
    var texture: MTLTexture?
        
    init(device: MTLDevice, textureURL: URL = transparentURL) {
        super.init(device: device)
        
        let textureLoader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false
        ]
        do {
            texture = try textureLoader.newTexture(URL: textureURL, options: options)
        } catch {
            print("Warning: Couldn't load texture: \(error)")
        }
    }
    
    convenience init(device: MTLDevice, textureURL: URL = transparentURL, sphere: Sphere,
                    reflectance: Float = 0.0, refraction: Float = 1.0, opacity: Float = 1.0) {
        self.init(device: device, textureURL: textureURL)
        self.reflectance = reflectance
        self.refraction = refraction
        self.opacity = opacity
        
        addSphere(sphere: sphere)
        
        uploadToBuffers()
    }
    
    func addSphere(sphere: Sphere) {
        spheres.append(sphere)
    }
    
    func addSphere(origin: SIMD3<Float>, radius: Float, color: SIMD3<Float>) {
        let sphere = Sphere(origin: origin, radius: radius, color: color)
        spheres.append(sphere)
    }
    
    override func uploadToBuffers() {
        let options = getManagedBufferStorageMode()
        
        guard !spheres.isEmpty else {
            print("Warning: No spheres to upload")
            return
        }
        
        sphereBuffer = device.makeBuffer(
            bytes: &spheres,
            length: spheres.count * MemoryLayout<Sphere>.stride,
            options: options
        )
        
        var boundingBoxes: [MTLAxisAlignedBoundingBox] = []
        for sphere in spheres {
            let boundingBox = MTLAxisAlignedBoundingBox(
                min: MTLPackedFloat3Make(
                    sphere.origin.x - sphere.radius,
                    sphere.origin.y - sphere.radius,
                    sphere.origin.z - sphere.radius
                ),
                max: MTLPackedFloat3Make(
                    sphere.origin.x + sphere.radius,
                    sphere.origin.y + sphere.radius,
                    sphere.origin.z + sphere.radius
                )
            )
            boundingBoxes.append(boundingBox)
        }
        
        boundingBoxBuffer = device.makeBuffer(
            bytes: &boundingBoxes,
            length: boundingBoxes.count * MemoryLayout<MTLAxisAlignedBoundingBox>.stride,
            options: options
        )
        
        reflectanceBuffer = device.makeBuffer(
            bytes: &reflectance,
            length: MemoryLayout<Float>.stride,
            options: options
        )
        
        refractionBuffer = device.makeBuffer(
            bytes: &refraction,
            length: MemoryLayout<Float>.stride,
            options: options
        )
        
        opacityBuffer = device.makeBuffer(
            bytes: &opacity,
            length: MemoryLayout<Float>.stride,
            options: options
        )
        
        #if !os(iOS)
        sphereBuffer?.didModifyRange(0..<sphereBuffer!.length)
        boundingBoxBuffer?.didModifyRange(0..<boundingBoxBuffer!.length)
        reflectanceBuffer?.didModifyRange(0..<reflectanceBuffer!.length)
        refractionBuffer?.didModifyRange(0..<refractionBuffer!.length)
        opacityBuffer?.didModifyRange(0..<opacityBuffer!.length)
        #endif
    }
    
    override func geometryDescriptor() -> MTLAccelerationStructureGeometryDescriptor? {
        guard let boundingBoxBuffer = boundingBoxBuffer, !spheres.isEmpty else {
            print("Warning: No bounding box buffer or spheres")
            return nil
        }
        
        let descriptor = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
        descriptor.boundingBoxBuffer = boundingBoxBuffer
        descriptor.boundingBoxCount = spheres.count
        
        return descriptor
    }
    
    override func resources() -> [MTLResource] {
        var resourceArray: [MTLResource] = []
        
        if let sb = sphereBuffer { resourceArray.append(sb) }
        if let rb = reflectanceBuffer { resourceArray.append(rb) }
        if let rb = refractionBuffer { resourceArray.append(rb) }
        if let ob = opacityBuffer { resourceArray.append(ob) }
        if let tex = texture { resourceArray.append(tex) }
        
        return resourceArray
    }
    
    func getTexture() -> MTLTexture? { // fix later if you ever will
        return texture
    }

    override func intersectionFunctionName() -> String? {
        return "sphereIntersectionFunction"
    }
}
