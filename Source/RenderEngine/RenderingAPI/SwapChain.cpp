#include "SwapChain.h"

#include <array>
#include <iostream>
#include <climits>
#include <stdexcept>

#include "VulkanDefines.h"
#include "Logs/LogSystem.h"

SwapChain::SwapChain(VulkanHandler& device_ref, VkExtent2D window_extent)
    : device{device_ref}, window_extent{window_extent}
{
    initializeSwapChain();
}

void SwapChain::initializeSwapChain()
{
    createSwapChain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createSyncObjects();
}

SwapChain::SwapChain(VulkanHandler& device_ref, VkExtent2D window_extent, std::shared_ptr<SwapChain> previous)
    : device{device_ref}, window_extent{window_extent}, old_swap_chain{std::move(previous)}
{
    initializeSwapChain();

    old_swap_chain.reset();
}

SwapChain::~SwapChain()
{
    for (auto imageView: swap_chain_image_views)
    {
        vkDestroyImageView(device.getDeviceHandle(), imageView, nullptr);
    }
    swap_chain_image_views.clear();

    if (swap_chain != nullptr)
    {
        vkDestroySwapchainKHR(device.getDeviceHandle(), swap_chain, nullptr);
        swap_chain = nullptr;
    }

    for (auto framebuffer: swap_chain_framebuffers)
    {
        vkDestroyFramebuffer(device.getDeviceHandle(), framebuffer, nullptr);
    }

    vkDestroyRenderPass(device.getDeviceHandle(), render_pass, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device.getDeviceHandle(), render_finished_semaphores[i], nullptr);
        vkDestroySemaphore(device.getDeviceHandle(), image_available_semaphores[i], nullptr);
        vkDestroyFence(device.getDeviceHandle(), in_flight_fences[i], nullptr);
    }
}

VkResult SwapChain::acquireNextImage(uint32_t* image_index)
{
    vkWaitForFences(
        device.getDeviceHandle(),
        1,
        &in_flight_fences[current_frame],
        VK_TRUE,
        UINT_MAX);

    VkResult result = vkAcquireNextImageKHR(
        device.getDeviceHandle(),
        swap_chain,
        UINT_MAX,
        image_available_semaphores[current_frame],
        VK_NULL_HANDLE,
        image_index);

    return result;
}

VkResult SwapChain::submitCommandBuffers(const VkCommandBuffer* buffers, uint32_t buffers_count, uint32_t* image_index)
{
    if (images_in_flight[*image_index] != VK_NULL_HANDLE)
    {
        vkWaitForFences(device.getDeviceHandle(), 1, &images_in_flight[*image_index], VK_TRUE, UINT64_MAX);
    }
    images_in_flight[*image_index] = in_flight_fences[current_frame];

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::array<VkSemaphore, 1> wait_semaphores = {image_available_semaphores[current_frame]};
    std::array<VkPipelineStageFlags, 1> wait_stages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = wait_semaphores.size();
    submit_info.pWaitSemaphores = wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_stages.data();

    submit_info.commandBufferCount = buffers_count;
    submit_info.pCommandBuffers = buffers;

    std::array<VkSemaphore, 1> signal_semaphores = {render_finished_semaphores[current_frame]};
    submit_info.signalSemaphoreCount = signal_semaphores.size();
    submit_info.pSignalSemaphores = signal_semaphores.data();

    vkResetFences(device.getDeviceHandle(), 1, &in_flight_fences[current_frame]);
    if (vkQueueSubmit(device.getLogicalDevice().getGraphicsQueue(), 1, &submit_info, in_flight_fences[current_frame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    present_info.waitSemaphoreCount = signal_semaphores.size();
    present_info.pWaitSemaphores = signal_semaphores.data();

    std::array<VkSwapchainKHR, 1> swap_chains = {swap_chain};
    present_info.swapchainCount = swap_chains.size();
    present_info.pSwapchains = swap_chains.data();

    present_info.pImageIndices = image_index;

    auto result = vkQueuePresentKHR(device.getLogicalDevice().getPresentQueue(), &present_info);

    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    return result;
}

void SwapChain::createSwapChain()
{
    SwapChainSupportDetails swap_chain_support = device.getPhysicalDevice().querySwapChainSupport(device.getPhysicalDeviceHandle());

    VkSurfaceFormatKHR surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
    VkPresentModeKHR present_mode = chooseSwapPresentMode(swap_chain_support.presentModes);
    VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
    {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = device.getSurfaceKHRHandle();

    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = device.getPhysicalDevice().getQueueFamilyIndices();
    std::array<uint32_t, 2> queue_family_indices = {indices.graphicsFamily, indices.presentFamily};

    if (indices.graphicsFamily != indices.presentFamily)
    {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = queue_family_indices.size();
        create_info.pQueueFamilyIndices = queue_family_indices.data();
    } else
    {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;      // Optional
        create_info.pQueueFamilyIndices = nullptr;  // Optional
    }

    create_info.preTransform = swap_chain_support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;

    create_info.oldSwapchain = old_swap_chain == nullptr ? VK_NULL_HANDLE : old_swap_chain->swap_chain;

    if (vkCreateSwapchainKHR(device.getDeviceHandle(), &create_info, nullptr, &swap_chain) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device.getDeviceHandle(), swap_chain, &image_count, nullptr);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device.getDeviceHandle(), swap_chain, &image_count, swap_chain_images.data());

    swap_chain_image_format = surface_format.format;
    swap_chain_extent = extent;
}

void SwapChain::createRenderPass()
{
    VkAttachmentDescription color_attachment{};
    color_attachment.format = getSwapChainImageFormat();
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcAccessMask = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstSubpass = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 1> attachments = {color_attachment};
    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    if (vkCreateRenderPass(device.getDeviceHandle(), &render_pass_info, nullptr, &render_pass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create updateElements pass!");
    }
}

void SwapChain::createFramebuffers()
{
    swap_chain_framebuffers.resize(imageCount());
    for (size_t i = 0; i < imageCount(); i++)
    {
        std::array<VkImageView, 1> attachments = {swap_chain_image_views[i]};

        VkExtent2D extent = getSwapChainExtent();
        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = render_pass;
        framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebuffer_info.pAttachments = attachments.data();
        framebuffer_info.width = extent.width;
        framebuffer_info.height = extent.height;
        framebuffer_info.layers = 1;

        if (vkCreateFramebuffer(
            device.getDeviceHandle(),
            &framebuffer_info,
            nullptr,
            &swap_chain_framebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void SwapChain::createImageViews()
{
    swap_chain_image_views.resize(swap_chain_images.size());
    for (size_t i = 0; i < swap_chain_images.size(); i++)
    {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = swap_chain_images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = swap_chain_image_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device.getDeviceHandle(), &view_info, nullptr, &swap_chain_image_views[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image view!");
        }
    }
}

void SwapChain::createSyncObjects()
{
    image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
    images_in_flight.resize(imageCount(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device.getDeviceHandle(), &semaphore_info, VulkanDefines::NO_CALLBACK, &image_available_semaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device.getDeviceHandle(), &semaphore_info, VulkanDefines::NO_CALLBACK, &render_finished_semaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device.getDeviceHandle(), &fence_info, VulkanDefines::NO_CALLBACK, &in_flight_fences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats)
{
    for (const auto& available_format: available_formats)
    {
        if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return available_format;
        }
    }

    return available_formats[0];
}

VkPresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available_present_modes)
{
    for (const auto& available_present_mode: available_present_modes)
    {
        if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            LogSystem::log(LogSeverity::LOG, "Present mode: Mailbox");
            return available_present_mode;
        }
    }

    LogSystem::log(LogSeverity::LOG, "Present mode: V-Sync");
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    VkExtent2D actual_extent = window_extent;
    actual_extent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actual_extent.width));
    actual_extent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actual_extent.height));

    return actual_extent;
}
