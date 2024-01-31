#pragma once

#include "Device.h"

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

class SwapChain
{
public:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    SwapChain(Device& device_ref, VkExtent2D window_extent);
    ~SwapChain();

    SwapChain(const SwapChain&) = delete;
    SwapChain& operator=(const SwapChain&) = delete;

    VkFramebuffer getFrameBuffer(int index) { return swap_chain_framebuffers[index]; }
    VkRenderPass getRenderPass() { return render_pass; }
    VkImageView getImageView(int index) { return swap_chain_image_views[index]; }
    size_t imageCount() { return swap_chain_images.size(); }
    VkFormat getSwapChainImageFormat() { return swap_chain_image_format; }
    VkExtent2D getSwapChainExtent() { return swap_chain_extent; }
    uint32_t width() { return swap_chain_extent.width; }
    uint32_t height() { return swap_chain_extent.height; }

    float extentAspectRatio() { return static_cast<float>(swap_chain_extent.width) / static_cast<float>(swap_chain_extent.height); }
    VkFormat findDepthFormat();

    VkResult acquireNextImage(uint32_t* image_index);
    VkResult submitCommandBuffers(const VkCommandBuffer* buffers, uint32_t* image_index);

private:
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available_present_modes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    VkFormat swap_chain_image_format;
    VkExtent2D swap_chain_extent;

    std::vector<VkFramebuffer> swap_chain_framebuffers;
    VkRenderPass render_pass;

    std::vector<VkImage> depth_images;
    std::vector<VkDeviceMemory> depth_image_memorys;
    std::vector<VkImageView> depth_image_views;
    std::vector<VkImage> swap_chain_images;
    std::vector<VkImageView> swap_chain_image_views;

    Device& device;
    VkExtent2D window_extent;

    VkSwapchainKHR swapchain;

    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence> in_flight_fences;
    std::vector<VkFence> images_in_flight;
    size_t current_frame = 0;
};
