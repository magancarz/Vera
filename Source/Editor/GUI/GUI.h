#pragma once

#include <Editor/Window/Window.h>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "RenderEngine/RenderingAPI/Descriptors/DescriptorPool.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/FrameInfo.h"
#include "Editor/GUI/Components/GUIContainer.h"

class GUI
{
public:
    GUI(VulkanHandler& device, Window& window, SwapChain* swap_chain);
    ~GUI();

    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

    void updateGUIElements(FrameInfo& frame_info);
    void renderGUIElements(VkCommandBuffer command_buffer);

private:
    VulkanHandler& device;
    Window& window;
    SwapChain* swap_chain;

    void initializeGUIComponents();

    std::unique_ptr<GUIContainer> root_component;

    void initializeImGui();
    void createContext();

    void createDescriptorPool();

    std::unique_ptr<DescriptorPool> descriptor_pool;

    void createRenderPass();
    VkAttachmentDescription createAttachmentDescription();
    VkAttachmentReference createColorAttachment();
    VkSubpassDescription createSubpass(VkAttachmentReference* color_attachment);
    VkSubpassDependency createSubpassDependency();

    VkRenderPass render_pass;

    void setupRendererBackends();

    void createFramebuffers();

    std::vector<VkFramebuffer> framebuffers;

    void startNewFrame();
};