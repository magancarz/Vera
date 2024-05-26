#pragma once

#include "VeraMaterial.h"
#include "Assets/AssetManager.h"

class WavefrontMaterial : public VeraMaterial
{
public:
    WavefrontMaterial(const std::unique_ptr<MemoryAllocator>& memory_allocator, const MaterialInfo& material_info, std::string material_name, std::shared_ptr<Texture> texture);
};
