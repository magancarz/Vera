#include <iostream>
#include "GUI.h"

#include "imgui_impl_vulkan.h"
#include "imgui_impl_glfw.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "RenderEngine/FrameInfo.h"
#include "Editor/GUI/Components/GUIContainer.h"
#include "Editor/GUI/Components/SceneSettingsWindow.h"
#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Logs/LogSystem.h"

GUI::GUI(VulkanHandler& device, Window& window, SwapChain* swap_chain)
    : device{device}, window{window}, swap_chain{swap_chain}
{
    initializeImGui();
    initializeGUIComponents();
}

void GUI::initializeImGui()
{
    createContext();
    createDescriptorPool();
    createRenderPass();
    createFramebuffers();
    setupRendererBackends();
}

void GUI::createContext()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();
}

void GUI::createDescriptorPool()
{
    descriptor_pool = DescriptorPool::Builder(device)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
            .setMaxSets(1)
            .setPoolFlags(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
            .build();
}

void GUI::createRenderPass()
{
    VkAttachmentDescription attachment = createAttachmentDescription();
    VkAttachmentReference color_attachment = createColorAttachment();
    VkSubpassDescription subpass = createSubpass(&color_attachment);
    VkSubpassDependency dependency = createSubpassDependency();

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;
    if (vkCreateRenderPass(device.getDeviceHandle(), &info, VulkanDefines::NO_CALLBACK, &render_pass) != VK_SUCCESS)
    {
        throw std::runtime_error("Could not create Dear ImGui's updateElements pass");
    }
}

VkAttachmentDescription GUI::createAttachmentDescription()
{
    VkAttachmentDescription attachment{};
    attachment.format = swap_chain->getSwapChainImageFormat();
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    return attachment;
}

VkAttachmentReference GUI::createColorAttachment()
{
    VkAttachmentReference color_attachment{};
    color_attachment.attachment = 0;
    color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    return color_attachment;
}

VkSubpassDescription GUI::createSubpass(VkAttachmentReference* color_attachment)
{
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = color_attachment;

    return subpass;
}

VkSubpassDependency GUI::createSubpassDependency()
{
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;  // or VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    return dependency;
}

void GUI::createFramebuffers()
{
    std::array<VkImageView, 1> attachment{};
    VkFramebufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = render_pass;
    info.attachmentCount = 1;
    info.pAttachments = attachment.data();
    info.width = swap_chain->width();
    info.height = swap_chain->height();
    info.layers = 1;

    framebuffers.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkImageView backbuffer_view = swap_chain->getImageView(i);
        attachment[0] = backbuffer_view;
        if (vkCreateFramebuffer(device.getDeviceHandle(), &info, VulkanDefines::NO_CALLBACK, &framebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Couldn't create framebuffer!");
        }
    }
}

namespace callbacks
{
    void checkResult(VkResult result)
    {
        if (result == 0)
        {
            return;
        }

        LogSystem::log(LogSeverity::ERROR, "[vulkan] Error: VkResult = ", result);
        if (result < 0)
        {
            abort();
        }
    }
}

void GUI::setupRendererBackends()
{
    if (auto as_glfw_window = dynamic_cast<GLFWWindow*>(&WindowSystem::get()))
    {
        ImGui_ImplGlfw_InitForVulkan(as_glfw_window->getGFLWwindow(), true);
    }
    else
    {
        LogSystem::log(LogSeverity::FATAL, "Unable to fetch glfw window while initializing imgui!");
    }

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = device.getInstanceHandle();
    init_info.PhysicalDevice = device.getPhysicalDeviceHandle();
    init_info.Device = device.getDeviceHandle();
    init_info.QueueFamily = device.getPhysicalDevice().getQueueFamilyIndices().graphicsFamily;
    init_info.Queue = device.getLogicalDevice().getGraphicsQueue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = descriptor_pool->descriptorPool();
    init_info.Subpass = 0;
    init_info.MinImageCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    init_info.ImageCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = callbacks::checkResult;
    ImGui_ImplVulkan_Init(&init_info, render_pass);
}

void GUI::initializeGUIComponents()
{
    root_component = std::make_unique<GUIContainer>("Root Component");
    root_component->addComponent(std::make_unique<SceneSettingsWindow>());
}

GUI::~GUI()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    for (auto& framebuffer : framebuffers)
    {
        vkDestroyFramebuffer(device.getDeviceHandle(), framebuffer, VulkanDefines::NO_CALLBACK);
    }
    vkDestroyRenderPass(device.getDeviceHandle(), render_pass, VulkanDefines::NO_CALLBACK);
}

void GUI::updateGUIElements(FrameInfo& frame_info)
{
    startNewFrame();
    root_component->update(frame_info);
}

void GUI::startNewFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::renderGUIElements(VkCommandBuffer command_buffer)
{
    ImGui::Render();

    uint32_t frame_index = swap_chain->getCurrentFrameIndex();
    VkClearValue clear_value{};
    clear_value.color.float32[0] = 0.f;
    clear_value.color.float32[1] = 0.f;
    clear_value.color.float32[2] = 0.f;
    clear_value.color.float32[3] = 1.f;
    VkRenderPassBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = render_pass;
    info.framebuffer = framebuffers[frame_index];
    info.renderArea.extent.width = swap_chain->width();
    info.renderArea.extent.height = swap_chain->height();
    info.clearValueCount = 1;
    info.pClearValues = &clear_value;
    vkCmdBeginRenderPass(command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
    vkCmdEndRenderPass(command_buffer);
}
