#include "AssetManager.h"

#include "RenderEngine/Models/OBJModel.h"
#include "RenderEngine/Materials/MaterialBuilder.h"

AssetManager::AssetManager(VulkanFacade* vulkan_facade)
    : vulkan_facade{vulkan_facade} {}

void AssetManager::loadNeededAssetsForProject(const ProjectInfo& project_info)
{
    printf("Loading needed assets for the project %s...\n", project_info.project_name.c_str());
    for (auto& object_info : project_info.objects_infos)
    {
        if (!fetchModel(object_info.model_name) || !fetchMaterial(object_info.material_name))
        {
            printf("Loading needed assets for project %s ended in failure\n", project_info.project_name.c_str());
            return;
        }
    }

    printf("Loading needed assets for project %s ended in success\n", project_info.project_name.c_str());
}

std::shared_ptr<Model> AssetManager::fetchModel(const std::string& model_name)
{
    assert(vulkan_facade && "Vulkan facade can't be nullptr");

    printf("Fetching model named %s\n", model_name.c_str());
    if (available_models.contains(model_name))
    {
        printf("Model found in available models list. Returning...\n");
        return available_models[model_name];
    }

    printf("Model was not found in available models list. Loading from file...\n");
    std::shared_ptr<Model> new_model = OBJModel::createModelFromFile(*vulkan_facade, model_name);
    available_models[model_name] = new_model;
    return new_model;
}

std::shared_ptr<Material> AssetManager::fetchMaterial(const std::string& material_name)
{
    assert(vulkan_facade && "Vulkan facade can't be nullptr");

    printf("Fetching material named %s\n", material_name.c_str());
    if (available_materials.contains(material_name))
    {
        printf("Material found in available materials list. Returning...\n");
        return available_materials[material_name];
    }

    printf("Material was not found in available materials list. Loading from file...\n");
    std::shared_ptr<Material> new_material = MaterialBuilder(*vulkan_facade).fromAssetFile(material_name).build();
    available_materials[material_name] = new_material;
    return new_material;
}

void AssetManager::clearResources()
{
    available_models.clear();
    available_materials.clear();
}