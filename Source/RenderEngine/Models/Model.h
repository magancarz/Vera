#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/Memory/Buffer.h"
#include "ModelDescription.h"

class Model
{
public:
    explicit Model(std::string model_name, std::string required_material = "");
    virtual ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    virtual void bind(VkCommandBuffer command_buffer);
    virtual void draw(VkCommandBuffer command_buffer) const;

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] bool isMaterialRequired() const { return !required_material.empty(); }
    [[nodiscard]] virtual std::vector<std::string> getRequiredMaterials() const { return {required_material}; }
    [[nodiscard]] virtual std::vector<ModelDescription> getModelDescriptions() const;

protected:
    std::string name{};
    std::string required_material{};

    std::unique_ptr<Buffer> vertex_buffer;
    uint32_t vertex_count{0};

    std::unique_ptr<Buffer> index_buffer;
    uint32_t index_count{0};
};
