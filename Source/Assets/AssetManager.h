#pragma once

#include <memory>
#include <unordered_map>
#include <mutex>

#include "RenderEngine/RenderingAPI/Device.h"
#include "Project/Project.h"
#include "RenderEngine/Models/Model.h"

class AssetManager
{
public:
    AssetManager(const AssetManager&) = delete;
    AssetManager operator=(const AssetManager&) = delete;

    ~AssetManager() = default;

    bool loadNeededAssetsForProject(const ProjectInfo& project_info);

    std::shared_ptr<Model> fetchModel(const std::string& model_name);
    std::shared_ptr<Material> fetchMaterial(const std::string& material_name);

    static std::shared_ptr<AssetManager> get(Device* device = nullptr)
    {
        assert(instance || device && "Device can't be nullptr while instance hasn't been created yet!");

        std::lock_guard<std::mutex> lock(mutex);
        if (!instance)
        {
            instance = std::shared_ptr<AssetManager>(new AssetManager(*device));
        }

        return instance;
    }

private:
    explicit AssetManager(Device& device);

    inline static std::shared_ptr<AssetManager> instance;
    inline static std::mutex mutex;

    Device& device;

    std::unordered_map<std::string, std::shared_ptr<Model>> available_models;
    std::unordered_map<std::string, std::shared_ptr<Material>> available_materials;
};
