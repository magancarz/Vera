#pragma once

#include "Assets/AssetManager.h"

class MockAssetManager : public AssetManager
{
public:
    MockAssetManager(std::unique_ptr<MemoryAllocator>& memory_allocator);

    std::shared_ptr<Model> fetchModel(const std::string& model_name) override;
    std::shared_ptr<Material> fetchMaterial(const std::string& material_name) override;
};
