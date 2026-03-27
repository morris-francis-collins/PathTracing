//
//  ContentView.swift
//  PathTracing
//

import SwiftUI
import MetalKit

struct ContentView: View {
    @State private var rendererType: RendererType = .megaKernel
    @State private var debugType: DebugType = Color
    @State private var widthText: String = ""
    @State private var heightText: String = ""
    @State private var currentPixelWidth: Int = 0
    @State private var currentPixelHeight: Int = 0
    @State private var metalViewPointSize: CGSize = .zero
    @State private var isProgrammaticResize: Bool = false
    
    @State private var device: MTLDevice = MTLCreateSystemDefaultDevice()!
    @State private var scene: GameScene? = nil
    
    private let hudWidth: CGFloat = 180
    
    var body: some View {
        HStack(spacing: 0) {
            GeometryReader { geometry in
                if let scene = scene {
                    MetalView(device: device, scene: scene,
                              rendererType: rendererType, debugType: debugType)
                        // No .id() — the view persists, only the renderer swaps
                        .onChange(of: geometry.size) { oldSize, newSize in
                            guard !isProgrammaticResize,
                                  newSize.width > 0, newSize.height > 0 else { return }
                            metalViewPointSize = newSize
                            let (pw, ph) = physicalPixels(from: newSize)
                            currentPixelWidth = pw
                            currentPixelHeight = ph
                            widthText = "\(pw)"
                            heightText = "\(ph)"
                        }
                        .onAppear {
                            guard geometry.size.width > 0, geometry.size.height > 0 else { return }
                            metalViewPointSize = geometry.size
                            let (pw, ph) = physicalPixels(from: geometry.size)
                            currentPixelWidth = pw
                            currentPixelHeight = ph
                            widthText = "\(pw)"
                            heightText = "\(ph)"
                        }
                } else {
                    ProgressView("Loading scene…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            
            VStack(alignment: .leading, spacing: 16) {
                rendererSection
                if rendererType == .debug {
                    debugTypeSection
                }
                Divider()
                resolutionSection
                Spacer()
            }
            .padding(12)
            .frame(width: hudWidth)
            .background(Color(NSColor.controlBackgroundColor))
        }
        .onAppear {
            if scene == nil {
                scene = GameScene(device: device)
            }
        }
    }
        
    private var rendererSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Renderer")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("", selection: $rendererType) {
                Text(RendererType.megaKernel.displayName).tag(RendererType.megaKernel)
                Text(RendererType.waveFront.displayName).tag(RendererType.waveFront)
                Text(RendererType.debug.displayName).tag(RendererType.debug)
            }
            .labelsHidden()
            .pickerStyle(.menu)
        }
    }
    
    private var debugTypeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Debug Channel")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("", selection: $debugType) {
                Text("Color").tag(Color)
                Text("Roughness").tag(Roughness)
                Text("Metallic").tag(Metallic)
                Text("IOR").tag(IOR)
                Text("Transmission").tag(Transmission)
                Text("Clearcoat").tag(Clearcoat)
                Text("Clearcoat Roughness").tag(ClearcoatRoughness)
                Text("Thickness").tag(ThicknessFactor)
                Text("Attenuation Color").tag(AttenuationColor)
                Text("Attenuation Distance").tag(AttenuationDistance)
                Text("Alpha").tag(Alpha)
                Text("Alpha Mode").tag(AlphaMode)
                Text("Emission").tag(Emission)
                Text("BXDF").tag(BXDF)
                Text("Normal").tag(Normal)
            }
            .labelsHidden()
            .pickerStyle(.menu)
        }
    }
    
    private var resolutionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Resolution")
                .font(.caption)
                .foregroundColor(.secondary)
            
            HStack(spacing: 4) {
                TextField("W", text: $widthText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 55)
                    .onSubmit { applyResolution() }
                
                Text("×")
                    .foregroundColor(.secondary)
                
                TextField("H", text: $heightText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 55)
                    .onSubmit { applyResolution() }
            }
            
            HStack(spacing: 8) {
                Button("Apply") { applyResolution() }
                    .controlSize(.small)
            }
            
            Text("\(currentPixelWidth) × \(currentPixelHeight) px")
                .font(.caption.monospacedDigit())
                .foregroundColor(.secondary)
        }
    }
    
    private func physicalPixels(from viewSize: CGSize) -> (Int, Int) {
        let scale = screenScaleFactor()
        return (Int(round(viewSize.width * scale)), Int(round(viewSize.height * scale)))
    }
    
    private func screenScaleFactor() -> CGFloat {
        NSScreen.main?.backingScaleFactor ?? 2.0
    }

    private func applyResolution() {
        guard let w = Int(widthText), let h = Int(heightText),
              w > 0, h > 0, w <= 8192, h <= 8192 else { return }
        
        guard let window = NSApplication.shared.mainWindow,
              currentPixelWidth > 0, currentPixelHeight > 0 else { return }
        
        isProgrammaticResize = true
        
        let scale = screenScaleFactor()
        let deltaWidth  = CGFloat(w - currentPixelWidth)  / scale
        let deltaHeight = CGFloat(h - currentPixelHeight) / scale
        
        currentPixelWidth = w
        currentPixelHeight = h
        metalViewPointSize = CGSize(width: CGFloat(w) / scale, height: CGFloat(h) / scale)
        
        let newFrame = NSRect(
            x: window.frame.origin.x,
            y: window.frame.origin.y - deltaHeight,
            width: window.frame.width + deltaWidth,
            height: window.frame.height + deltaHeight
        )
        
        window.setFrame(newFrame, display: true, animate: true)
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            isProgrammaticResize = false
        }
    }
}

// MARK: - MetalView

struct MetalView: NSViewRepresentable {
    let device: MTLDevice
    let scene: GameScene
    var rendererType: RendererType
    var debugType: DebugType
    
    class Coordinator {
        let device: MTLDevice
        let scene: GameScene
        var renderer: Renderer?
        var activeRendererType: RendererType?
        var activeDebugType: DebugType?
        weak var mtkView: KeyHandlingMTKView?
        
        init(device: MTLDevice, scene: GameScene) {
            self.device = device
            self.scene = scene
        }
        
        func swapRenderer(to type: RendererType, debugType: DebugType) {
            mtkView?.isPaused = true
            
            let newRenderer: Renderer
            switch type {
            case .megaKernel:
                newRenderer = MegaKernelRenderer(device: device, scene: scene)
            case .waveFront:
                newRenderer = WaveFrontRenderer(device: device, scene: scene)
            case .debug:
                newRenderer = DebugRenderer(device: device, scene: scene, debugType: debugType)
            }
            
            renderer = newRenderer
            activeRendererType = type
            activeDebugType = debugType
            
            mtkView?.delegate = newRenderer
            mtkView?.renderer = newRenderer
            
            if let mtkView = mtkView {
                let size = mtkView.drawableSize
                if size.width > 0, size.height > 0 {
                    newRenderer.mtkView(mtkView, drawableSizeWillChange: size)
                }
            }
            
            mtkView?.isPaused = false
        }
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(device: device, scene: scene)
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = KeyHandlingMTKView()
        mtkView.preferredFramesPerSecond = 60
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.isPaused = false
        mtkView.depthStencilPixelFormat = .depth32Float
        
        context.coordinator.mtkView = mtkView
        context.coordinator.swapRenderer(to: rendererType, debugType: debugType)
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        let needsSwap =
            context.coordinator.activeRendererType != rendererType ||
            (rendererType == .debug && context.coordinator.activeDebugType != debugType)
        
        if needsSwap {
            context.coordinator.swapRenderer(to: rendererType, debugType: debugType)
        }
    }
}

enum RendererType: Equatable {
    case megaKernel
    case waveFront
    case debug
    
    var displayName: String {
        switch self {
        case .megaKernel: return "Mega Kernel"
        case .waveFront: return "Wave Front"
        case .debug: return "Debug"
        }
    }
}

extension NSWindow {
    var titlebarHeight: CGFloat {
        frame.height - contentRect(forFrameRect: frame).height
    }
}
