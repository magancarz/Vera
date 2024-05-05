#pragma once

#include "RenderEngine/RenderingAPI/Device.h"
#include "RenderEngine/Window.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"

class GUI
{
public:
    GUI(Device& device, Window& window, std::shared_ptr<SwapChain> swap_chain);
    ~GUI();

    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

    void render(FrameInfo& frame_info);

private:
    Device& device;
    Window& window;
    std::shared_ptr<SwapChain> swap_chain;

    void initializeImGui();
    void createContext();

    void createDescriptorPool();

    std::unique_ptr<DescriptorPool> descriptor_pool;

    void createRenderPass();

    VkRenderPass render_pass;

    void setupRendererBackends();

    void createFramebuffers();

    std::vector<VkFramebuffer> framebuffers;

    void startFrame();
    void endFrame(VkCommandBuffer command_buffer);
};