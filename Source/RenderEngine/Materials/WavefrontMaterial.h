#pragma once

#include "VeraMaterial.h"

class WavefrontMaterial : public VeraMaterial
{
public:
    static std::shared_ptr<WavefrontMaterial> fromAssetFile(VulkanFacade& vulkan_facade, const std::string& asset_name);

    WavefrontMaterial(VulkanFacade& vulkan_facade, const MaterialInfo& material_info, std::string material_name, std::shared_ptr<Texture> texture);
};
