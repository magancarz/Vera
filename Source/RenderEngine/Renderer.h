#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"

struct SimplePushConstantData
{
    glm::mat2 transform{1.f};
    glm::vec2 offset;
    alignas(16) glm::vec3 color;
};

class Renderer
{
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void renderScene();

private:
    void loadObjects();
    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers();
    void freeCommandBuffers();
    void recreateSwapChain();
    void recordCommandBuffer(int image_index);
    void renderObjects(VkCommandBuffer command_buffer);

    Device device;
    std::unique_ptr<SwapChain> swap_chain;
    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<Object> objects;
};
