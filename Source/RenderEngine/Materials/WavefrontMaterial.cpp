#include "WavefrontMaterial.h"

std::shared_ptr<WavefrontMaterial> WavefrontMaterial::fromAssetFile(VulkanFacade& vulkan_facade, const std::string& asset_name)
{
    MaterialInfo material_info{};

    return std::make_shared<WavefrontMaterial>(vulkan_facade, material_info, asset_name, nullptr);
}

WavefrontMaterial::WavefrontMaterial(
        VulkanFacade& vulkan_facade,
        const MaterialInfo& material_info,
        std::string material_name,
        std::shared_ptr<Texture> texture)
    : VeraMaterial(vulkan_facade, material_info, std::move(material_name), std::move(texture)) {}
