#include "MockAssetManager.h"
#include "MockModel.h"
#include "MockMemoryAllocator.h"

MockAssetManager::MockAssetManager(std::unique_ptr<MemoryAllocator>& memory_allocator)
    : AssetManager(nullptr, memory_allocator) {}

std::shared_ptr<Model> MockAssetManager::fetchModel(const std::string& model_name)
{
    if (!available_models.contains(model_name))
    {
        available_models.emplace(model_name, std::make_shared<MockModel>(model_name));
    }

    return available_models.at(model_name);
}

std::shared_ptr<Material> MockAssetManager::fetchMaterial(const std::string& material_name)
{
    constexpr static MaterialInfo simple_info{};
    if (!available_materials.contains(material_name))
    {
        available_materials.emplace(material_name, std::make_shared<Material>(simple_info, material_name, nullptr));
    }

    return available_materials.at(material_name);
}
