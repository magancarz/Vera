#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"

class Renderer
{
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void renderScene();

private:
    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers();
    void drawFrame();

    Device device;
    SwapChain swap_chain{device, Display::getExtent()};
    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;
    std::vector<VkCommandBuffer> command_buffers;
};
