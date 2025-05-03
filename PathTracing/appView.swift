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
                .frame(width: 800, height: 600)
        }
    }
}
