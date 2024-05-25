#pragma once

#include "Assets/AssetManager.h"
#include "OBJModel.h"

class OBJModelLoader
{
public:
    static std::shared_ptr<OBJModel> createFromFile(VulkanFacade& vulkan_facade, AssetManager* asset_manager, const std::string& model_name);
};
