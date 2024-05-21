#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/Window.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/FrameInfo.h"
#include "Editor/GUI/Components/Container.h"

class GUI
{
public:
    GUI(VulkanFacade& device, Window& window, std::shared_ptr<SwapChain> swap_chain);
    ~GUI();

    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

    void updateGUIElements(FrameInfo& frame_info);
    void renderGUIElements(VkCommandBuffer command_buffer);

private:
    VulkanFacade& device;
    Window& window;
    std::shared_ptr<SwapChain> swap_chain;

    void initializeGUIComponents();

    std::shared_ptr<Container> root_component;

    void initializeImGui();
    void createContext();

    void createDescriptorPool();

    std::unique_ptr<DescriptorPool> descriptor_pool;

    void createRenderPass();

    VkRenderPass render_pass;

    void setupRendererBackends();

    void createFramebuffers();

    std::vector<VkFramebuffer> framebuffers;

    void startNewFrame();
};