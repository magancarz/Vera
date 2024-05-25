#pragma once

#include "Model.h"
#include "Assets/AssetManager.h"
#include "OBJModelInfo.h"

class OBJModel : public Model
{
public:
    OBJModel(VulkanFacade& device, const std::vector<OBJModelInfo>& obj_models_info, std::string name);
    OBJModel(VulkanFacade& device, const OBJModelInfo& obj_model_info);

    OBJModel(const OBJModel&) = delete;
    OBJModel& operator=(const OBJModel&) = delete;

    void bind(VkCommandBuffer command_buffer) override;
    void draw(VkCommandBuffer command_buffer) const override;

    [[nodiscard]] std::vector<std::string> getRequiredMaterials() const override;
    [[nodiscard]] std::vector<ModelDescription> getModelDescriptions() const override;

private:
    VulkanFacade& device;

    void createManyModels(const std::vector<OBJModelInfo>& obj_models_info);
    void createModel(const OBJModelInfo& model_info);
    void createVertexBuffers(const std::vector<Vertex>& vertices);
    void createIndexBuffers(const std::vector<uint32_t>& indices);

    std::vector<std::shared_ptr<Model>> models;
};
