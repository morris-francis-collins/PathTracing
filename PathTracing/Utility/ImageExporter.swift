//
//  ImageExporter.swift
//  PathTracing
//
//  Created on 3/27/26.
//

import ImageIO
import UniformTypeIdentifiers
import MetalKit

extension Renderer {
    func exportImage(format: ImageExportFormat) {
        guard let texture = finalImage else { return }

        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

        var pixels = [Float](repeating: 0, count: pixelCount * 4)
        texture.getBytes(&pixels,
                         bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                         from: MTLRegionMake2D(0, 0, width, height),
                         mipmapLevel: 0)

        let rowFloats = width * 4
        for y in 0..<(height / 2) {
            let topStart = y * rowFloats
            let botStart = (height - 1 - y) * rowFloats
            for x in 0..<rowFloats {
                pixels.swapAt(topStart + x, botStart + x)
            }
        }

        let panel = NSSavePanel()
        panel.canCreateDirectories = true

        switch format {
        case .exr:
            panel.allowedContentTypes = [UTType(filenameExtension: "exr")!]
            panel.nameFieldStringValue = "render.exr"
        case .png:
            panel.allowedContentTypes = [.png]
            panel.nameFieldStringValue = "render.png"
        }

        guard panel.runModal() == .OK, let url = panel.url else { return }

        switch format {
        case .exr:
            writeEXR(pixels: pixels, width: width, height: height, url: url)
        case .png:
            writePNG(pixels: pixels, width: width, height: height, url: url)
        }
    }

    private func writeEXR(pixels: [Float], width: Int, height: Int, url: URL) {
        let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB)!
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, "com.ilm.openexr-image" as CFString, 1, nil
        ) else { return }

        let bitmapInfo = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.premultipliedLast.rawValue |
            CGBitmapInfo.floatComponents.rawValue |
            CGBitmapInfo.byteOrder32Little.rawValue)

        guard let provider = CGDataProvider(data: Data(
            bytes: pixels, count: pixels.count * MemoryLayout<Float>.size
        ) as CFData),
        let image = CGImage(
            width: width, height: height,
            bitsPerComponent: 32, bitsPerPixel: 128,
            bytesPerRow: width * 4 * MemoryLayout<Float>.size,
            space: colorSpace, bitmapInfo: bitmapInfo,
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        ) else { return }

        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        print("Exported EXR: \(url.path)")
    }

    private func writePNG(pixels: [Float], width: Int, height: Int, url: URL) {
        var srgbPixels = [UInt8](repeating: 0, count: width * height * 4)
        
        for i in 0..<(width * height) {
            let r = pixels[i * 4 + 0]
            let g = pixels[i * 4 + 1]
            let b = pixels[i * 4 + 2]

            srgbPixels[i * 4 + 0] = floatToSRGB8(r)
            srgbPixels[i * 4 + 1] = floatToSRGB8(g)
            srgbPixels[i * 4 + 2] = floatToSRGB8(b)
            srgbPixels[i * 4 + 3] = 255
        }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let context = CGContext(
            data: &srgbPixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        let image = context.makeImage(),
        let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { return }

        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        print("Exported PNG: \(url.path)")
    }
}

func reinhardTonemap(_ x : Float) -> Float {
    return x / (1.0 + x)
}

func floatToSRGB8(_ x: Float) -> UInt8 {
    return UInt8(min(max(reinhardTonemap(x), 0.0), 1.0) * 255.0 + 0.5)
}

enum ImageExportFormat {
    case exr
    case png
}
