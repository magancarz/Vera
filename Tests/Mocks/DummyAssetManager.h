#pragma once

#include <string>

#include "Assets/AssetManager.h"

class DummyAssetManager : public AssetManager
{
public:
    DummyAssetManager(VulkanHandler& vulkan_facade, MemoryAllocator& memory_allocator);

    Mesh* getMeshIfAvailable(const std::string& mesh_name) const;
    Model* getModelIfAvailable(const std::string& model_name) const;
    Material* getMaterialIfAvailable(const std::string& material_name) const;
    DeviceTexture* getTextureIfAvailable(const std::string& texture_name) const;
};
