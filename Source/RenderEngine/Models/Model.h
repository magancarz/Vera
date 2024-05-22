#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/RenderingAPI/Vertex.h"
#include "RenderEngine/RenderingAPI/Buffer.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/BlasInstance.h"
#include "RenderEngine/Models/ObjectDescription.h"
#include "ModelDescription.h"

class Model
{
public:
    Model(std::string model_name);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void bind(VkCommandBuffer command_buffer);
    void draw(VkCommandBuffer command_buffer) const;

    [[nodiscard]] std::string getName() const { return name; }
    [[nodiscard]] ModelDescription getModelDescription() const;

protected:
    std::string name;

    std::unique_ptr<Buffer> vertex_buffer;
    uint32_t vertex_count{0};

    std::unique_ptr<Buffer> index_buffer;
    uint32_t index_count{0};
};
