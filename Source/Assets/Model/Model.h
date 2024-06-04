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

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] bool isMaterialRequired() const { return !required_material.empty(); }
    [[nodiscard]] std::string getRequiredMaterial() const { return required_material; }

    [[nodiscard]] ModelDescription getModelDescription() const;

protected:
    std::string name{};
    std::string required_material{};

    std::unique_ptr<Buffer> vertex_buffer;
    uint32_t vertex_count{0};

    std::unique_ptr<Buffer> index_buffer;
    uint32_t index_count{0};
};
