#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"

class Renderer
{
public:
    Renderer(Window& window, Device& device, World& world);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(FrameInfo& frame_info);

private:
    Window& window;
    Device& device;
    World& world;

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

    void recreateSwapChain();
    void createCommandBuffers();

    void createSceneRenderer();

    std::unique_ptr<SceneRenderer> scene_renderer;

    VkCommandBuffer beginFrame();
    void endFrame();

    void freeCommandBuffers();

    std::unique_ptr<SwapChain> swap_chain;
    std::vector<VkCommandBuffer> command_buffers;

    uint32_t current_image_index{0};
    bool is_frame_started{false};
    int current_frame_index{0};
};
