//
//  ContentView.swift
//  PathTracing
//

import SwiftUI
import MetalKit
import UniformTypeIdentifiers
import ImageIO

struct ContentView: View {
    @State private var rendererType: RendererType = .megaKernel
    @State private var debugType: DebugType = Color
    @State private var widthText: String = ""
    @State private var heightText: String = ""
    @State private var currentPixelWidth: Int = 0
    @State private var currentPixelHeight: Int = 0
    @State private var metalViewPointSize: CGSize = .zero
    @State private var isProgrammaticResize: Bool = false
    @State private var coordinator: MetalView.Coordinator?
    
    @State private var device: MTLDevice = MTLCreateSystemDefaultDevice()!
    @State private var scene: GameScene? = nil
    
    @State private var comparisonEnabled = false
    @State private var referencePixels: [Float]? = nil
    @State private var referenceWidth: Int = 0
    @State private var referenceHeight: Int = 0
    @State private var referencePath: String = ""
    @State private var displayMode: DisplayMode = .render
    @State private var comparisonStats: ComparisonStats?
    @State private var statsTimer: Timer?
    @State private var dimensionMismatch = false
    
    private let hudWidth: CGFloat = 200
    
    var body: some View {
        HStack(spacing: 0) {
            GeometryReader { geometry in
                if let scene = scene {
                    MetalView(device: device, scene: scene,
                              rendererType: rendererType, debugType: debugType,
                              coordinator: $coordinator)
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
                if rendererType == .debug { debugTypeSection }
                Divider()
                resolutionSection
                Divider()
                exportSection
                Divider()
                comparisonSection
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
                Text("×").foregroundColor(.secondary)
                TextField("H", text: $heightText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 55)
                    .onSubmit { applyResolution() }
            }
            Button("Apply") { applyResolution() }
                .controlSize(.small)
            Text("\(currentPixelWidth) × \(currentPixelHeight) px")
                .font(.caption.monospacedDigit())
                .foregroundColor(.secondary)
        }
    }
    
    private var exportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Export")
                .font(.caption)
                .foregroundColor(.secondary)
            HStack(spacing: 8) {
                Button("EXR") { coordinator?.renderer?.exportImage(format: .exr) }
                    .controlSize(.small)
                Button("PNG") { coordinator?.renderer?.exportImage(format: .png) }
                    .controlSize(.small)
            }
        }
    }
    
    private var comparisonSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Comparison", isOn: $comparisonEnabled)
                .font(.caption)
                .onChange(of: comparisonEnabled) { _, enabled in
                    if !enabled {
                        statsTimer?.invalidate()
                        statsTimer = nil
                        coordinator?.renderer?.displayMode = 0
                        coordinator?.renderer?.referenceTexture = nil
                        referencePixels = nil
                        comparisonStats = nil
                        referencePath = ""
                        displayMode = .render
                        dimensionMismatch = false
                    }
                }

            if comparisonEnabled {
                Button("Load Reference EXR") { loadReferenceEXR() }
                    .controlSize(.small)

                if dimensionMismatch {
                    Text("Resolution mismatch: ref is \(referenceWidth)×\(referenceHeight)")
                        .font(.caption)
                        .foregroundColor(.red)
                }

                if !referencePath.isEmpty {
                    Text(referencePath)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                if referencePixels != nil && !dimensionMismatch {
                    Picker("Display", selection: $displayMode) {
                        ForEach(DisplayMode.allCases, id: \.self) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .onChange(of: displayMode) { _, newMode in
                        coordinator?.renderer?.displayMode = UInt32(newMode.rawValue)
                    }

                    HStack(spacing: 8) {
                        Button("Update") { updateStats() }
                            .controlSize(.small)
                        Button(statsTimer != nil ? "Stop" : "Auto") {
                            if statsTimer != nil {
                                statsTimer?.invalidate()
                                statsTimer = nil
                            } else {
                                statsTimer = Timer.scheduledTimer(
                                    withTimeInterval: 2.0, repeats: true
                                ) { _ in updateStats() }
                            }
                        }
                        .controlSize(.small)
                    }

                    if let stats = comparisonStats {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(String(format: "MSE:     %.2e", stats.mse))
                            Text(String(format: "relMSE:  %.2e", stats.relMSE))
                            Text(String(format: "PSNR:    %.1f dB", stats.psnr))
                            Text(String(format: "PSNR99:  %.1f dB", stats.clippedPSNR))
                            Text(String(format: "p50:     %.2e", stats.median))
                            Text(String(format: "p99:     %.4f", stats.p99))
                            Text(String(format: "p999:    %.4f", stats.p999))
                            Text(String(format: "max:     %.4f", stats.maxError))
                            Text(String(format: "SMAPE:   %.4f", stats.smape))
                        }
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                    }
                }
            }
        }
    }
    
    private func loadReferenceEXR() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "exr")!]
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .origin: MTKTextureLoader.Origin.flippedVertically
        ]
        
        guard let tex = try? loader.newTexture(URL: url, options: options) else {
            print("Failed to load reference EXR")
            return
        }

        let width = tex.width
        let height = tex.height
        referenceWidth = width
        referenceHeight = height
        referencePath = url.lastPathComponent

        guard let renderer = coordinator?.renderer, let accumTex = renderer.finalImage, accumTex.width == width, accumTex.height == height else {
            dimensionMismatch = true
            return
        }
        
        dimensionMismatch = false

        var pixels = [Float](repeating: 0, count: width * height * 4)
        tex.getBytes(&pixels,
                     bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                     from: MTLRegionMake2D(0, 0, width, height),
                     mipmapLevel: 0)

        referencePixels = pixels

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: tex.pixelFormat, width: width, height: height, mipmapped: false)
        desc.usage = .shaderRead
        guard let flippedTex = device.makeTexture(descriptor: desc) else { return }
        pixels.withUnsafeBytes { ptr in
            flippedTex.replace(region: MTLRegionMake2D(0, 0, width, height),
                               mipmapLevel: 0,
                               withBytes: ptr.baseAddress!,
                               bytesPerRow: width * 4 * MemoryLayout<Float>.size)
        }

        coordinator?.renderer?.referenceTexture = flippedTex
        coordinator?.renderer?.displayMode = 0
        displayMode = .render
        comparisonStats = nil
    }
    
    private func updateStats() {
        guard let renderer = coordinator?.renderer,
              let accumTex = renderer.finalImage,
              let refPixels = referencePixels,
              accumTex.width == referenceWidth,
              accumTex.height == referenceHeight else { return }

        comparisonStats = computeComparisonStats(
            renderTex: accumTex,
            referencePixels: refPixels,
            width: referenceWidth,
            height: referenceHeight
        )
    }

    private func physicalPixels(from viewSize: CGSize) -> (Int, Int) {
        let scale = screenScaleFactor()
        return (Int(round(viewSize.width * scale)), Int(round(viewSize.height * scale)))
    }
    
    private func screenScaleFactor() -> CGFloat {
        NSScreen.main?.backingScaleFactor ?? 2.0
    }

    private func applyResolution() {
        guard let w = Int(widthText), let h = Int(heightText), w > 0, h > 0, w <= 8192, h <= 8192 else { return }
        guard let window = NSApplication.shared.mainWindow, currentPixelWidth > 0, currentPixelHeight > 0 else { return }

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
    let device: MTLDevice
    let scene: GameScene
    var rendererType: RendererType
    var debugType: DebugType
    
    @Binding var coordinator: MetalView.Coordinator?
    
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
        
        DispatchQueue.main.async {
            coordinator = context.coordinator
        }
        
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

struct ComparisonStats {
    var mse: Float = 0
    var rmse: Float = 0
    var psnr: Float = 0
    var relMSE: Float = 0
    var maxError: Float = 0
    var median: Float = 0
    var p99: Float = 0
    var p999: Float = 0
    var clippedMSE: Float = 0
    var clippedPSNR: Float = 0
    var smape: Double = 0
}

func computeComparisonStats(renderTex: MTLTexture, referencePixels: [Float], width: Int, height: Int) -> ComparisonStats {
    var renderPixels = [Float](repeating: 0, count: width * height * 4)
    renderTex.getBytes(&renderPixels,
                       bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                       from: MTLRegionMake2D(0, 0, width, height),
                       mipmapLevel: 0)

    let pixelCount = width * height
    var errors = [Float](repeating: 0, count: pixelCount)
    var totalSE: Double = 0
    var totalRelSE: Double = 0
    var maxErr: Float = 0
    var totalSMAPE: Double = 0

    for i in 0..<pixelCount {
        let ri = i * 4
        let rr = renderPixels[ri + 0], rg = renderPixels[ri + 1], rb = renderPixels[ri + 2]
        let fr = referencePixels[ri + 0], fg = referencePixels[ri + 1], fb = referencePixels[ri + 2]
        let dr = abs(rr - fr), dg = abs(rg - fg), db = abs(rb - fb)
        let sr = abs(rr) + abs(fr), sg = abs(rg) + abs(fg), sb = abs(rb) + abs(fb)
        
        let se = Double(dr * dr + dg * dg + db * db) / 3.0
        let err = sqrt(Float(se))
        errors[i] = err
        totalSE += se
        maxErr = max(maxErr, err)
        
        let smape = (Double(dr / max(sr, 1e-4)) + Double(dg / max(sg, 1e-4)) + Double(db / max(sb, 1e-4))) / 3.0
        totalSMAPE += smape

        let refLum = Double(referencePixels[ri] + referencePixels[ri+1] + referencePixels[ri+2]) / 3.0
        totalRelSE += se / (refLum * refLum)
    }

    errors.sort()

    let p50 = errors[pixelCount / 2]
    let p99 = errors[Int(Float(pixelCount) * 0.99)]
    let p999 = errors[Int(Float(pixelCount) * 0.999)]

    let clipIdx = Int(Float(pixelCount) * 0.999)
    var clippedSE: Double = 0
    for i in 0..<clipIdx {
        clippedSE += Double(errors[i] * errors[i])
    }
    let clippedMSE = Float(clippedSE / Double(clipIdx))

    let mse = Float(totalSE / Double(pixelCount))

    return ComparisonStats(
        mse: mse,
        rmse: sqrt(mse),
        psnr: mse > 0 ? -10.0 * log10(mse) : 100.0,
        relMSE: Float(totalRelSE / Double(pixelCount)),
        maxError: maxErr,
        median: p50,
        p99: p99,
        p999: p999,
        clippedMSE: clippedMSE,
        clippedPSNR: clippedMSE > 0 ? -10.0 * log10(clippedMSE) : 100.0,
        smape: totalSMAPE / Double(pixelCount)
    )
}

enum DisplayMode: Int, CaseIterable {
    case render = 0
    case reference = 1
    case falseColor = 2

    var label: String {
        switch self {
        case .render: return "Render"
        case .reference: return "Reference"
        case .falseColor: return "Error"
        }
    }
}
