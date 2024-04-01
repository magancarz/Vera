#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/Systems/SimpleRenderSystem.h"
#include "RenderEngine/Systems/PointLightSystem.h"

class Renderer
{
public:
    Renderer(Window& window, Device& device);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(FrameInfo& frame_info);

    std::vector<std::shared_ptr<Material>> getAvailableMaterials() { return materials; }

private:
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

    void createCommandBuffers();
    void freeCommandBuffers();
    void recreateSwapChain();

    void createDescriptorPool();

    Window& window;
    Device& device;
    std::unique_ptr<SwapChain> swap_chain;
    std::vector<VkCommandBuffer> command_buffers;
    std::unique_ptr<DescriptorPool> global_pool{};

    uint32_t current_image_index{0};
    bool is_frame_started{false};
    int current_frame_index{0};

    std::vector<std::unique_ptr<Buffer>> ubo_buffers;
    std::vector<VkDescriptorSet> global_uniform_buffer_descriptor_sets;

    std::unique_ptr<SimpleRenderSystem> simple_render_system;
    std::unique_ptr<PointLightSystem> point_light_render_system;

    std::vector<std::shared_ptr<Material>> materials;
};
