#pragma once

#include "Assets/AssetManager.h"
#include "OBJModel.h"

class OBJModelLoader
{
public:
    static std::shared_ptr<OBJModel> createFromFile(const std::unique_ptr<MemoryAllocator>& memory_allocator, AssetManager* asset_manager, const std::string& model_name);
};
