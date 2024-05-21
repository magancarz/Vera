#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/RenderingAPI/Vertex.h"
#include "RenderEngine/RenderingAPI/Buffer.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/Models/AccelerationStructure.h"
#include "RenderEngine/Models/BlasInstance.h"
#include "RenderEngine/Models/ObjectDescription.h"

class Model
{
public:
    struct Builder
    {
        std::vector<Vertex> vertices{};
        std::vector<uint32_t> indices{};
        float area;

        void loadModel(const std::string& filepath);
    };

    Model(VulkanFacade& device, const Model::Builder& builder);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    static std::unique_ptr<Model> createModelFromFile(VulkanFacade& device, const std::string& model_name);

    void bind(VkCommandBuffer command_buffer);
    void draw(VkCommandBuffer command_buffer);

    std::unique_ptr<Buffer> vertex_buffer;
    std::unique_ptr<Buffer> index_buffer;

    BlasInstance createBlasInstance(const glm::mat4& transform, uint32_t id);
    void getModelDescription(ObjectDescription& object_description) const;

private:
    VulkanFacade& device;

    float surface_area{0.f};

    void createVertexBuffers(const std::vector<Vertex>& vertices);

    uint32_t vertex_count;

    void createIndexBuffers(const std::vector<uint32_t>& indices);

    bool has_index_buffer{false};
    uint32_t index_count;

    void createBlas();

    AccelerationStructure blas{};
};
