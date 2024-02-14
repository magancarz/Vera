#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "FrameInfo.h"

struct SimplePushConstantData
{
    glm::mat4 model{1.f};
    glm::mat4 normal_matrix{1.f};
};

class SimpleRenderSystem
{
public:
    SimpleRenderSystem(
            Device& device, Window& window, VkRenderPass render_pass, VkDescriptorSetLayout global_set_layout);
    ~SimpleRenderSystem();

    SimpleRenderSystem(const SimpleRenderSystem&) = delete;
    SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

    void renderObjects(
        FrameInfo& frame_info,
        std::vector<Object>& objects);

private:
    void createPipelineLayout(VkDescriptorSetLayout global_set_layout);
    void createPipeline(VkRenderPass render_pass);

    Device& device;
    Window& window;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
};