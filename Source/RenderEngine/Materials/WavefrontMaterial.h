#pragma once

#include "VeraMaterial.h"
#include "Assets/AssetManager.h"

class WavefrontMaterial : public VeraMaterial
{
public:
    WavefrontMaterial(VulkanFacade& vulkan_facade, const MaterialInfo& material_info, std::string material_name, std::shared_ptr<Texture> texture);
};
