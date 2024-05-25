#include "WavefrontMaterial.h"

#include "Assets/AssetManager.h"

WavefrontMaterial::WavefrontMaterial(
        VulkanFacade& vulkan_facade,
        const MaterialInfo& material_info,
        std::string material_name,
        std::shared_ptr<Texture> texture)
    : VeraMaterial(vulkan_facade, material_info, std::move(material_name), std::move(texture)) {}
