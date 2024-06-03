#include "WavefrontMaterial.h"

#include "Assets/AssetManager.h"

WavefrontMaterial::WavefrontMaterial(
        MemoryAllocator& memory_allocator,
        const MaterialInfo& material_info,
        std::string material_name,
        Texture* diffuse_texture,
        Texture* normal_texture)
    : VeraMaterial(memory_allocator, material_info, std::move(material_name), diffuse_texture, normal_texture) {}
