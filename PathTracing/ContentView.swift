//
//  ContentView.swift
//  PathTracing
//

import SwiftUI
import MetalKit

struct ContentView: NSViewRepresentable {
    @EnvironmentObject var gameScene: GameScene
    let device: MTLDevice = MTLCreateSystemDefaultDevice()!
    
    func makeCoordinator() -> Renderer {
        Renderer(device: device, scene: gameScene)
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = KeyHandlingMTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.drawableSize = mtkView.frame.size
        mtkView.isPaused = false
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.renderer = context.coordinator
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
}
