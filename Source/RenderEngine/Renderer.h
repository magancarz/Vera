#pragma once

#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "Assets/Model/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"
#include "Editor/GUI/GUI.h"
#include "RenderEngine/PostProcessing/PostProcessing.h"
#include "RenderEngine/SceneRenderers/RayTraced/RayTracedRenderer.h"

class Renderer
{
public:
    Renderer(Window& window, VulkanFacade& device, MemoryAllocator& memory_allocator, World& world, AssetManager& asset_manager);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(FrameInfo& frame_info);

private:
    Window& window;
    VulkanFacade& device;
    MemoryAllocator& memory_allocator;
    World& world;
    AssetManager& asset_manager;

    [[nodiscard]] VkCommandBuffer getCurrentCommandBuffer() const
    {
        assert(is_frame_started && "Cannot get command buffer when frame not in progress");
        return command_buffers[current_frame_index];
    }

    [[nodiscard]] int getFrameIndex() const
    {
        assert(is_frame_started && "Cannot get frame index when frame not in progress!");
        return current_frame_index;
    }

    void recreateSwapChain();

    std::unique_ptr<SwapChain> swap_chain;

    void createGUI();

    std::unique_ptr<GUI> gui;

    void createCommandBuffers();

    void createSceneRenderer();

    std::unique_ptr<RayTracedRenderer> scene_renderer;

    void createPostProcessingStage();

    std::unique_ptr<DescriptorPool> post_process_texture_descriptor_pool;
    std::unique_ptr<DescriptorSetLayout> post_process_texture_descriptor_set_layout;
    VkDescriptorSet post_process_texture_descriptor_set_handle{VK_NULL_HANDLE};
    std::unique_ptr<PostProcessing> post_processing;

    VkCommandBuffer beginFrame();
    void endFrame();

    void applyPostProcessing(FrameInfo& frame_info);
    void beginSwapChainRenderPass(VkCommandBuffer command_buffer);
    void endSwapChainRenderPass(VkCommandBuffer command_buffer);

    void freeCommandBuffers();

    std::vector<VkCommandBuffer> command_buffers;

    bool is_frame_started{false};
    int current_frame_index{0};
    uint32_t current_image_index{0};
};
