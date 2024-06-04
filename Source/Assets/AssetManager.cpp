#include "AssetManager.h"

#include "Logs/LogSystem.h"
#include "OBJ/OBJLoader.h"
#include "RenderEngine/Materials/VeraMaterial.h"

AssetManager::AssetManager(VulkanFacade& vulkan_facade, MemoryAllocator& memory_allocator)
    : vulkan_facade{vulkan_facade}, memory_allocator{memory_allocator} {}

void AssetManager::loadAssetsRequiredForProject(const ProjectInfo& project_info)
{
    LogSystem::log(LogSeverity::LOG, "Loading needed assets for the project ", project_info.project_name.c_str(), "...");
    for (auto& object_info : project_info.objects_infos)
    {
        if (!fetchModel(object_info.mesh_name) || !fetchMaterial(object_info.material_name))
        {
            LogSystem::log(LogSeverity::FATAL, "Loading needed assets for project ", project_info.project_name.c_str(), " ended in failure");
            return;
        }
    }

    LogSystem::log(LogSeverity::LOG, "Loading needed assets for project ", project_info.project_name.c_str(), " ended in success");
}

Mesh* AssetManager::fetchMesh(const std::string& mesh_name)
{
    if (available_meshes.contains(mesh_name))
    {
        return available_meshes.at(mesh_name).get();
    }

    LogSystem::log(LogSeverity::FATAL, "Mesh ", mesh_name.c_str(), " not found in available meshes!");
    return nullptr;
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
    return storeMaterial(VeraMaterial::fromAssetFile(memory_allocator, *this, material_name));
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

Texture* AssetManager::fetchTexture(const std::string& texture_name, VkFormat image_format)
{
    LogSystem::log(LogSeverity::LOG, "Fetching texture named ", texture_name.c_str());
    if (available_textures.contains(texture_name))
    {
        LogSystem::log(LogSeverity::LOG, "Texture found in available textures list. Returning...");
        return available_textures[texture_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Texture was not found in available textures list. Loading from file...");
    auto new_texture = std::make_unique<Texture>(vulkan_facade, memory_allocator, texture_name, image_format);
    available_textures[texture_name] = std::move(new_texture);
    return available_textures[texture_name].get();
}

Texture* AssetManager::storeTexture(std::unique_ptr<Texture> texture)
{
    assert(texture && "It is useless to store empty texture");
    auto texture_name = texture->getName();
    available_textures[texture_name] = std::move(texture);
    return available_textures[texture_name].get();
}

void AssetManager::clearResources()
{
    available_models.clear();
    available_materials.clear();
    available_textures.clear();
}