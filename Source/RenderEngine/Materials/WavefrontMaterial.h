#pragma once

#include "VeraMaterial.h"
#include "Assets/AssetManager.h"

class WavefrontMaterial : public VeraMaterial
{
public:
    WavefrontMaterial(
            MemoryAllocator& memory_allocator,
            const MaterialInfo& material_info,
            std::string material_name,
            Texture* diffuse_texture,
            Texture* normal_texture);
};
