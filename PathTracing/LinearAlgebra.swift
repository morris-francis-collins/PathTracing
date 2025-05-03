//
//  LinearAlgebra.swift
//  PathTracing
//
//  Created on 3/22/25.
//

import simd

class LinearAlgebra {
    
    static func identity() -> float4x4 {
        return float4x4(
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        )
    }
    
    static func translate(translation: simd_float3) -> float4x4 {
        return float4x4(
            [1,                0,             0,             0],
            [0,                1,             0,             0],
            [0,                0,             1,             0],
            [translation.x,    translation.y, translation.z, 1]
        )
    }
        
    static func createLookAt(eye: simd_float3, target: simd_float3, up: simd_float3) -> float4x4 {
        let forwards: simd_float3 = simd.normalize(target - eye)
        let right: simd_float3 = simd.normalize(simd.cross(forwards, up))
        let up2: simd_float3 = simd.normalize(simd.cross(right, forwards))
        
        return float4x4(
            [            right.x,             up2.x,             forwards.x,       0],
            [            right.y,             up2.y,             forwards.y,       0],
            [            right.z,             up2.z,             forwards.z,       0],
            [-simd.dot(right,eye), -simd.dot(up2,eye), -simd.dot(forwards,eye),    1]
        )
    }
    
    static func createPerpectiveProjection(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
        let f: Float = 1 / tan(fovY * .pi / 360)
        let A: Float = f / aspect
        let B: Float = f
        let C: Float = far / (far - near)
        let D: Float = 1
        let E: Float = -near * far / (far - near)
        
        return float4x4(
            [A, 0, 0, 0],
            [0, B, 0, 0],
            [0, 0, C, D],
            [0, 0, E, 0]
        )
    }
    
    static func rotate(eulers: simd_float3) -> float4x4 {
        return rotateX(θ: eulers.x) * rotateY(θ: eulers.y) * rotateZ(θ: eulers.z)
    }
    
    static func scale(scale: simd_float3) -> float4x4 {
        return matrix_float4x4(
            [scale.x, 0,       0,       0],
            [0,       scale.y, 0,       0],
            [0,       0,       scale.z, 0],
            [0,       0,       0,       1]
        )
    }
    
    static private func rotateX(θ: Float) -> float4x4 {
        return float4x4(
            [1,       0,      0, 0],
            [0,  cos(θ), sin(θ), 0],
            [0, -sin(θ), cos(θ), 0],
            [0,       0,      0, 1]
        )
    }
    
    static private func rotateY(θ: Float) -> float4x4 {
        return float4x4(
            [cos(θ), 0, -sin(θ), 0],
            [     0, 1,       0, 0],
            [sin(θ), 0,  cos(θ), 0],
            [     0, 0,       0, 1]
        )
    }
    
    static private func rotateZ(θ: Float) -> float4x4 {
        return float4x4(
            [ cos(θ), sin(θ), 0, 0],
            [-sin(θ), cos(θ), 0, 0],
            [      0,      0, 1, 0],
            [      0,      0, 0, 1]
        )
    }
    
    static func transformPosition(position: SIMD3<Float>, with transform: float4x4) -> SIMD3<Float> {
        let homogeneous = SIMD4<Float>(position, 1)
        let transformed = transform * homogeneous
        return SIMD3<Float>(transformed.x, transformed.y, transformed.z) / transformed.w
    }
    
    static func transformNormal(normal: SIMD3<Float>, with transform: float4x4) -> SIMD3<Float> {
        let t3x3 = float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        
        let normalMatrix = simd_transpose(simd_inverse(t3x3))
        let transformed = normalMatrix * normal
        return simd_normalize(transformed)
    }
}
