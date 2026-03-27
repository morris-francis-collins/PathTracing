//
//  ContentView.swift
//  PathTracing
//

import SwiftUI
import MetalKit

struct ContentView: View {
    @State private var rendererType: RendererType = .megaKernel
    @State private var widthText: String = ""
    @State private var heightText: String = ""
    @State private var currentPixelWidth: Int = 0
    @State private var currentPixelHeight: Int = 0
    @State private var metalViewPointSize: CGSize = .zero
    @State private var isProgrammaticResize: Bool = false
    
    private let hudWidth: CGFloat = 180
    
    var body: some View {
        HStack(spacing: 0) {
            // Metal render view — fills all remaining space
            GeometryReader { geometry in
                MetalView(rendererType: rendererType)
                    .id(rendererType)
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
            }
            
            VStack(alignment: .leading, spacing: 16) {
                rendererSection
                Divider()
                resolutionSection
                Spacer()
            }
            .padding(12)
            .frame(width: hudWidth)
            .background(Color(NSColor.controlBackgroundColor))
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

struct MetalView: NSViewRepresentable {
    let device: MTLDevice = MTLCreateSystemDefaultDevice()!
    var rendererType: RendererType
    
    func makeCoordinator() -> Renderer {
        switch rendererType {
        case .megaKernel:
            return MegaKernelRenderer(device: device, scene: GameScene(device: device))
        case .waveFront:
            return WaveFrontRenderer(device: device, scene: GameScene(device: device))
        }
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = KeyHandlingMTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.device = device
        mtkView.framebufferOnly = false
        mtkView.isPaused = false
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.renderer = context.coordinator
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
}

enum RendererType {
    case megaKernel
    case waveFront
    
    var displayName: String {
        switch self {
        case .megaKernel: return "Mega Kernel"
        case .waveFront: return "Wave Front"
        }
    }
}

extension NSWindow {
    var titlebarHeight: CGFloat {
        frame.height - contentRect(forFrameRect: frame).height
    }
}
