#include "DummyAssetManager.h"

DummyAssetManager::DummyAssetManager(VulkanHandler& vulkan_facade, MemoryAllocator& memory_allocator)
    : AssetManager(vulkan_facade, memory_allocator) {}

Mesh* DummyAssetManager::getMeshIfAvailable(const std::string& mesh_name) const
{
    if (available_meshes.contains(mesh_name))
    {
        return available_meshes.at(mesh_name).get();
    }

    return nullptr;
}

Model* DummyAssetManager::getModelIfAvailable(const std::string& model_name) const
{
    if (available_models.contains(model_name))
    {
        return available_models.at(model_name).get();
    }

    return nullptr;
}

Material* DummyAssetManager::getMaterialIfAvailable(const std::string& material_name) const
{
    if (available_materials.contains(material_name))
    {
        return available_materials.at(material_name).get();
    }

    return nullptr;
}

DeviceTexture* DummyAssetManager::getTextureIfAvailable(const std::string& texture_name) const
{
    if (available_textures.contains(texture_name))
    {
        return available_textures.at(texture_name).get();
    }

    return nullptr;
}
