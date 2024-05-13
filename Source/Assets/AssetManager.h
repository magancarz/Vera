#pragma once

#include <unordered_map>

#include "RenderEngine/RenderingAPI/Device.h"
#include "Project/Project.h"
#include "RenderEngine/Models/Model.h"

class AssetManager
{
public:
    AssetManager(Device& device);
    ~AssetManager() = default;

    bool loadNeededAssetsForProject(const ProjectInfo& project_info);

    std::shared_ptr<Model> fetchModel(const std::string& model_name);
    std::shared_ptr<Material> fetchMaterial(const std::string& material_name);

private:
    Device& device;

    std::unordered_map<std::string, std::shared_ptr<Model>> available_models;
    std::unordered_map<std::string, std::shared_ptr<Material>> available_materials;
};
