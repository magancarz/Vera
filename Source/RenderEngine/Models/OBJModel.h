#pragma once

#include "Model.h"
#include "Assets/AssetManager.h"

class OBJModel : public Model
{
public:
    struct Builder
    {
        std::vector<Vertex> vertices{};
        std::vector<uint32_t> indices{};

        void loadModel(const AssetManager* asset_manager, const std::string& filepath);
    };

    OBJModel(VulkanFacade& device, const OBJModel::Builder& builder, std::string model_name);

    OBJModel(const OBJModel&) = delete;
    OBJModel& operator=(const OBJModel&) = delete;

    static std::unique_ptr<Model> createModelFromFile(VulkanFacade& vulkan_facade, const AssetManager* asset_manager, const std::string& model_name);

private:
    VulkanFacade& device;

    void createVertexBuffers(const std::vector<Vertex>& vertices);
    void createIndexBuffers(const std::vector<uint32_t>& indices);
};
