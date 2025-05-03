//
//  PathTracing.swift
//  PathTracing
//

import SwiftUI

@main
struct ProjectApp: App {
    let device: MTLDevice
    @StateObject private var gameScene: GameScene

    init() {
        let createdDevice = MTLCreateSystemDefaultDevice()!
        self.device = createdDevice
        self._gameScene = StateObject(wrappedValue: GameScene(device: createdDevice))
    }

    var body: some Scene {
        WindowGroup {
            appView()
                .environmentObject(gameScene)
        }
    }
}
