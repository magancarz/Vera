#pragma once

#include <memory>
#include <unordered_map>
#include <mutex>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Project/Project.h"
#include "RenderEngine/Models/Model.h"

class AssetManager
{
public:
    AssetManager(const AssetManager&) = delete;
    AssetManager operator=(const AssetManager&) = delete;

    ~AssetManager() = default;

    void loadNeededAssetsForProject(const ProjectInfo& project_info);
    void clearResources();

    std::shared_ptr<Model> fetchModel(const std::string& model_name);
    std::shared_ptr<Material> fetchMaterial(const std::string& material_name);

    static std::shared_ptr<AssetManager> get(VulkanFacade* device = nullptr);

private:
    explicit AssetManager(VulkanFacade& device);

    inline static std::shared_ptr<AssetManager> instance;
    inline static std::mutex mutex;

    VulkanFacade& device;

    std::unordered_map<std::string, std::shared_ptr<Model>> available_models;
    std::unordered_map<std::string, std::shared_ptr<Material>> available_materials;
};
