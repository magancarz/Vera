#pragma once

#include "Material.h"
#include "Assets/AssetManager.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class VeraMaterial : public Material
{
public:
    static std::shared_ptr<VeraMaterial> fromAssetFile(const std::unique_ptr<MemoryAllocator>& memory_allocator, AssetManager* asset_manager, const std::string& asset_name);

    VeraMaterial(
            const std::unique_ptr<MemoryAllocator>& memory_allocator,
            const MaterialInfo& material_info,
            std::string material_name,
            std::shared_ptr<Texture> diffuse_texture,
            std::shared_ptr<Texture> normal_texture);

private:
    void createMaterialInfoBuffer(const std::unique_ptr<MemoryAllocator>& memory_allocator);
};
