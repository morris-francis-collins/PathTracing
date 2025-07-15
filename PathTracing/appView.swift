//
//  appView.swift
//  PathTracing
//

import SwiftUI
import MetalKit

struct appView: View {
    @EnvironmentObject var gamescene: GameScene
    
    var body: some View {
        VStack{
            Text("Path Tracing")
        
            ContentView()
                .frame(width: CGFloat(PIXEL_WIDTH), height: CGFloat(PIXEL_HEIGHT))
        }
    }
}
