#pragma once

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "Memory/Buffer.h"
#include "ModelDescription.h"

struct ModelInfo;
struct Vertex;
class MemoryAllocator;

class Model
{
public:
    Model(
        MemoryAllocator& memory_allocator,
        const ModelInfo& model_info);
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

    void createVertexBuffer(MemoryAllocator& memory_allocator, const std::vector<Vertex>& vertices);
    void createIndexBuffer(MemoryAllocator& memory_allocator, const std::vector<uint32_t>& indices);

    std::unique_ptr<Buffer> vertex_buffer;
    uint32_t vertex_count{0};

    std::unique_ptr<Buffer> index_buffer;
    uint32_t index_count{0};
};
