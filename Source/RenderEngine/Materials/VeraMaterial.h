#pragma once

#include "Material.h"

class VeraMaterial : public Material
{
public:
    static std::shared_ptr<VeraMaterial> fromAssetFile(VulkanFacade& vulkan_facade, const std::string& asset_name);

    VeraMaterial(VulkanFacade& vulkan_facade, const MaterialInfo& material_info, std::string material_name);

private:
    void createMaterialInfoBuffer(VulkanFacade& vulkan_facade);
    void assignMaterialHitGroupIndex();
};
