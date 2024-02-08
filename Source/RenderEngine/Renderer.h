#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "GUI/Display.h"

class Renderer
{
public:
    Renderer(Device& device);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    VkRenderPass getSwapChainRenderPass() const { return swap_chain->getRenderPass(); }
    bool isFrameInProgress() const { return is_frame_started; }

    VkCommandBuffer getCurrentCommandBuffer() const
    {
        assert(is_frame_started && "Cannot get command buffer when frame not in progress");
        return command_buffers[current_frame_index];
    }

    int getFrameIndex() const
    {
        assert(is_frame_started && "Cannot get frame index when frame not in progress!");
        return current_frame_index;
    }

    VkCommandBuffer beginFrame();
    void endFrame();
    void beginSwapChainRenderPass(VkCommandBuffer command_buffer);
    void endSwapChainRenderPass(VkCommandBuffer command_buffer);

private:
    void createCommandBuffers();
    void freeCommandBuffers();
    void recreateSwapChain();

    Device& device;
    std::unique_ptr<SwapChain> swap_chain;
    std::vector<VkCommandBuffer> command_buffers;

    uint32_t current_image_index{0};
    bool is_frame_started{false};
    int current_frame_index{0};
};
