#include "AssetManager.h"

#include "Logs/LogSystem.h"
#include "RenderEngine/Materials/VeraMaterial.h"
#include "RenderEngine/Models/OBJModelLoader.h"

AssetManager::AssetManager(VulkanFacade* vulkan_facade, std::unique_ptr<MemoryAllocator>& memory_allocator)
    : vulkan_facade{vulkan_facade}, memory_allocator{memory_allocator} {}

void AssetManager::loadNeededAssetsForProject(const ProjectInfo& project_info)
{
    LogSystem::log(LogSeverity::LOG, "Loading needed assets for the project ", project_info.project_name.c_str(), "...");
    for (auto& object_info : project_info.objects_infos)
    {
        if (!fetchModel(object_info.model_name) || !fetchMaterial(object_info.material_name))
        {
            LogSystem::log(LogSeverity::FATAL, "Loading needed assets for project ", project_info.project_name.c_str(), " ended in failure");
            return;
        }
    }

    LogSystem::log(LogSeverity::LOG, "Loading needed assets for project ", project_info.project_name.c_str(), " ended in success");
}

std::shared_ptr<Model> AssetManager::fetchModel(const std::string& model_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching model named ", model_name.c_str());
    if (available_models.contains(model_name))
    {
        LogSystem::log(LogSeverity::LOG, "Model found in available models list. Returning...");
        return available_models[model_name];
    }

    LogSystem::log(LogSeverity::LOG, "Model was not found in available models list. Loading from file...");
    std::shared_ptr<Model> new_model = OBJModelLoader::createFromFile(memory_allocator, this, model_name);
    available_models[model_name] = new_model;
    return new_model;
}

std::shared_ptr<Material> AssetManager::fetchMaterial(const std::string& material_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching material named ", material_name.c_str());
    if (available_materials.contains(material_name))
    {
        LogSystem::log(LogSeverity::LOG, "Material found in available materials list. Returning...");
        return available_materials[material_name];
    }

    LogSystem::log(LogSeverity::LOG, "Material was not found in available materials list. Loading from file...");
    std::shared_ptr<Material> new_material = VeraMaterial::fromAssetFile(memory_allocator, this, material_name);
    available_materials[material_name] = new_material;
    return new_material;
}

std::vector<std::shared_ptr<Material>> AssetManager::fetchRequiredMaterials(const std::shared_ptr<Model>& model)
{
    std::vector<std::string> required_materials = model->getRequiredMaterials();
    std::vector<std::shared_ptr<Material>> materials(required_materials.size());
    for (size_t i = 0; i < required_materials.size(); ++i)
    {
        materials[i] = fetchMaterial(required_materials[i]);
    }

    return materials;
}

std::shared_ptr<Texture> AssetManager::fetchTexture(const std::string& texture_name, VkFormat image_format)
{
    LogSystem::log(LogSeverity::LOG, "Fetching texture named ", texture_name.c_str());
    if (available_textures.contains(texture_name))
    {
        LogSystem::log(LogSeverity::LOG, "Texture found in available textures list. Returning...");
        return available_textures[texture_name];
    }

    LogSystem::log(LogSeverity::LOG, "Texture was not found in available textures list. Loading from file...");
    std::shared_ptr<Texture> new_texture = std::make_shared<Texture>(*vulkan_facade, memory_allocator, texture_name, image_format);
    available_textures[texture_name] = new_texture;
    return new_texture;
}

void AssetManager::clearResources()
{
    available_models.clear();
    available_materials.clear();
}