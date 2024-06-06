#include "AssetManager.h"

#include "Defines.h"
#include "Logs/LogSystem.h"
#include "OBJ/OBJLoader.h"
#include "Assets/Material/VeraMaterialLoader.h"
#include "Texture/TextureLoader.h"

AssetManager::AssetManager(VulkanHandler& vulkan_facade, MemoryAllocator& memory_allocator)
    : vulkan_facade{vulkan_facade}, memory_allocator{memory_allocator} {}

Mesh* AssetManager::fetchMesh(const std::string& mesh_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching mesh named ", mesh_name.c_str());
    if (available_meshes.contains(mesh_name))
    {
        LogSystem::log(LogSeverity::LOG, "Mesh found in available meshes list. Returning...");
        return available_meshes[mesh_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Mesh was not found in available meshes list. Loading from file...");
    OBJLoader::loadAssetsFromFile(memory_allocator, *this, mesh_name);
    return available_meshes[mesh_name].get();
}

Mesh* AssetManager::storeMesh(std::unique_ptr<Mesh> mesh)
{
    assert(mesh && "It is useless to store empty mesh");
    auto mesh_name = mesh->name;
    available_meshes[mesh_name] = std::move(mesh);
    return available_meshes[mesh_name].get();
}

Model* AssetManager::fetchModel(const std::string& model_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching model named ", model_name.c_str());
    if (available_models.contains(model_name))
    {
        LogSystem::log(LogSeverity::LOG, "Model found in available models list. Returning...");
        return available_models[model_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Model was not found in available models list. Loading from file...");
    OBJLoader::loadAssetsFromFile(memory_allocator, *this, model_name);
    return available_models[model_name].get();
}

Model* AssetManager::storeModel(std::unique_ptr<Model> model)
{
    assert(model && "It is useless to store empty model");
    auto model_name = model->getName();
    available_models[model_name] = std::move(model);
    return available_models[model_name].get();
}

Material* AssetManager::fetchMaterial(const std::string& material_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching material named ", material_name.c_str());
    if (available_materials.contains(material_name))
    {
        LogSystem::log(LogSeverity::LOG, "Material found in available materials list. Returning...");
        return available_materials[material_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Material was not found in available materials list. Loading from file...");
    VeraMaterialLoader::loadAssetFromFile(*this, material_name);
    return available_materials[material_name].get();
}

Material* AssetManager::storeMaterial(std::unique_ptr<Material> material)
{
    assert(material && "It is useless to store empty material");
    auto material_name = material->getName();
    available_materials[material_name] = std::move(material);
    return available_materials[material_name].get();
}

std::vector<Material*> AssetManager::fetchRequiredMaterials(const std::vector<std::string>& required_materials)
{
    std::vector<Material*> materials(required_materials.size());
    for (size_t i = 0; i < required_materials.size(); ++i)
    {
        materials[i] = fetchMaterial(required_materials[i]);
    }

    return materials;
}

DeviceTexture* AssetManager::fetchDiffuseTexture(const std::string& texture_name)
{
    return fetchTexture(texture_name, Assets::DIFFUSE_TEXTURE_FORMAT);
}

DeviceTexture* AssetManager::fetchNormalMap(const std::string& texture_name)
{
    return fetchTexture(texture_name, Assets::NORMAL_MAP_FORMAT);
}

DeviceTexture* AssetManager::fetchTexture(const std::string& texture_name, VkFormat format)
{
    LogSystem::log(LogSeverity::LOG, "Fetching texture named ", texture_name.c_str());
    if (available_textures.contains(texture_name))
    {
        LogSystem::log(LogSeverity::LOG, "Texture found in available textures list. Returning...");
        return available_textures[texture_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Texture was not found in available textures list. Loading from file...");
    return storeTexture(TextureLoader::loadFromAssetFile(vulkan_facade, memory_allocator, texture_name, format));
}

DeviceTexture* AssetManager::storeTexture(std::unique_ptr<DeviceTexture> texture)
{
    assert(texture && "It is useless to store empty texture");
    auto texture_name = texture->getName();
    available_textures[texture_name] = std::move(texture);
    return available_textures[texture_name].get();
}
