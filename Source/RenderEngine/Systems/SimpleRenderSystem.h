#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/FrameInfo.h"

struct SimplePushConstantData
{
    glm::mat4 model{1.f};
    glm::mat4 normal_matrix{1.f};
};

class SimpleRenderSystem
{
public:
    SimpleRenderSystem(
            Device& device,
            VkRenderPass render_pass,
            VkDescriptorSetLayout global_uniform_buffer_set_layout,
            VkDescriptorSetLayout global_texture_set_layout);
    ~SimpleRenderSystem();

    SimpleRenderSystem(const SimpleRenderSystem&) = delete;
    SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

    void renderObjects(FrameInfo& frame_info);

private:
    void createPipelineLayout(VkDescriptorSetLayout global_set_layout, VkDescriptorSetLayout global_texture_set_layout);
    void createPipeline(VkRenderPass render_pass);

    Device& device;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
};