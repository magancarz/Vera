#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"

struct SimplePushConstantData
{
    glm::mat4 transform{1.f};
    glm::mat4 normal_matrix{1.f};
};

class SimpleRenderSystem
{
public:
    SimpleRenderSystem(Device& device, Window& window, VkRenderPass render_pass);
    ~SimpleRenderSystem();

    SimpleRenderSystem(const SimpleRenderSystem&) = delete;
    SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

    void renderObjects(
        VkCommandBuffer command_buffer,
        std::vector<Object>& objects,
        const Camera& camera);

private:
    void createPipelineLayout();
    void createPipeline(VkRenderPass render_pass);

    Device& device;
    Window& window;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
};