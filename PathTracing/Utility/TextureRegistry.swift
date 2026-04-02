//
//  TextureRegistry.swift
//  PathTracing
//
//  Created on 4/29/25.
//

import MetalKit

class TextureRegistry {
    static let shared = TextureRegistry()
    
    private var textures: [MTLTexture] = []
    private var textureMap: [String: Int] = [:]
    
    func addTexture(_ texture: MTLTexture, identifier: String) -> Int {
        if let index = textureMap[identifier] {
            return index;
        }
                
        let index = textures.count
        textureMap[identifier] = index
        textures.append(texture)
        
        return index
    }
    
    func getIndex(for identifier: String) -> Int {
        if let index = textureMap[identifier] {
            return index
        }

        return 0
    }
    
    func getTextures() -> [MTLTexture] {
        return textures
    }
}
