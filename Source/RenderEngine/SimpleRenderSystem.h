#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "GUI/GUI.h"

struct SimplePushConstantData
{
    glm::mat4 transform{1.f};
    alignas(16) glm::vec3 color;
};

class SimpleRenderSystem
{
public:
    SimpleRenderSystem(Device& device, VkRenderPass render_pass);
    ~SimpleRenderSystem();

    SimpleRenderSystem(const SimpleRenderSystem&) = delete;
    SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

    void renderObjects(VkCommandBuffer command_buffer, std::vector<Object>& objects);

private:
    void createPipelineLayout();
    void createPipeline(VkRenderPass render_pass);

    Device& device;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
};