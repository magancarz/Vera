#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/FrameInfo.h"
#include "Editor/GUI/Components/GUIContainer.h"

class GUI
{
public:
    GUI(VulkanFacade& device, Window& window, SwapChain* swap_chain);
    ~GUI();

    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

    void updateGUIElements(FrameInfo& frame_info);
    void renderGUIElements(VkCommandBuffer command_buffer);

private:
    VulkanFacade& device;
    Window& window;
    SwapChain* swap_chain;

    void initializeGUIComponents();

    std::unique_ptr<GUIContainer> root_component;

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