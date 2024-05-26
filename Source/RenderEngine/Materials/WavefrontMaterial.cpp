#include "WavefrontMaterial.h"

#include "Assets/AssetManager.h"

WavefrontMaterial::WavefrontMaterial(
        const std::unique_ptr<MemoryAllocator>& memory_allocator,
        const MaterialInfo& material_info,
        std::string material_name,
        std::shared_ptr<Texture> texture)
    : VeraMaterial(memory_allocator, material_info, std::move(material_name), std::move(texture)) {}
