#pragma once

#include "Material.h"
#include "Assets/AssetManager.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class VeraMaterial : public Material
{
public:
    static std::unique_ptr<Material> fromAssetFile(MemoryAllocator& memory_allocator, AssetManager& asset_manager, const std::string& asset_name);

    VeraMaterial(
            MemoryAllocator& memory_allocator,
            const MaterialInfo& material_info,
            std::string material_name,
            Texture* diffuse_texture,
            Texture* normal_texture);

private:
    void createMaterialInfoBuffer(MemoryAllocator& memory_allocator);
};
