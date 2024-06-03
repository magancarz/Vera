#include "Renderer.h"

#include <thread>

#include "RenderEngine/SceneRenderers/RayTraced/RayTracedRenderer.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"

Renderer::Renderer(Window& window, VulkanFacade& device, MemoryAllocator& memory_allocator, World& world, AssetManager& asset_manager)
    : window{window}, device{device}, memory_allocator{memory_allocator}, world{world}, asset_manager(asset_manager)
{
    createCommandBuffers();
    recreateSwapChain();
    createGUI();
    createSceneRenderer();
    createPostProcessingStage();
}

void Renderer::createCommandBuffers()
{
    command_buffers.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool = device.getCommandPool();
    allocate_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

    if (vkAllocateCommandBuffers(device.getDevice(), &allocate_info, command_buffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate command buffers!");
    }
}

void Renderer::recreateSwapChain()
{
    auto extent = window.getExtent();
    while (extent.width == 0 || extent.height == 0)
    {
        extent = window.getExtent();
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(device.getDevice());

    if (swap_chain == nullptr)
    {
        swap_chain = std::make_unique<SwapChain>(device, extent);
    }
    else
    {
        std::shared_ptr<SwapChain> old_swap_chain = std::move(swap_chain);
        swap_chain = std::make_unique<SwapChain>(device, extent, old_swap_chain);

        if (!old_swap_chain->compareSwapFormats(*swap_chain))
        {
            throw std::runtime_error("Swap chain image or depth format has changed!");
        }
    }
}

void Renderer::createGUI()
{
    gui = std::make_unique<GUI>(device, window, swap_chain.get());
}

void Renderer::createSceneRenderer()
{
    scene_renderer = std::make_unique<RayTracedRenderer>(device, memory_allocator, asset_manager, world);
}

void Renderer::createPostProcessingStage()
{
    post_process_texture_descriptor_pool = DescriptorPool::Builder(device)
            .setMaxSets(SwapChain::MAX_FRAMES_IN_FLIGHT)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
            .build();

    post_process_texture_descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();

    VkDescriptorImageInfo ray_trace_image_descriptor_info{};
    ray_trace_image_descriptor_info.sampler = scene_renderer->getRayTracedImageSampler();
    ray_trace_image_descriptor_info.imageView = scene_renderer->getRayTracedImageViewHandle();
    ray_trace_image_descriptor_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    DescriptorWriter(*post_process_texture_descriptor_set_layout, *post_process_texture_descriptor_pool)
            .writeImage(0, &ray_trace_image_descriptor_info)
            .build(post_process_texture_descriptor_set_handle);

    post_processing = std::make_unique<PostProcessing>(
            device,
            asset_manager,
            swap_chain->getRenderPass(),
            post_process_texture_descriptor_set_layout->getDescriptorSetLayout());
}

Renderer::~Renderer()
{
    freeCommandBuffers();
}

void Renderer::freeCommandBuffers()
{
    vkFreeCommandBuffers(
        device.getDevice(),
        device.getCommandPool(),
        static_cast<uint32_t>(command_buffers.size()),
        command_buffers.data());
    command_buffers.clear();
}

void Renderer::render(FrameInfo& frame_info)
{
    if (auto command_buffer = beginFrame())
    {
        frame_info.command_buffer = command_buffer;
        frame_info.window_size = swap_chain->getSwapChainExtent();
        frame_info.ray_traced_texture = post_process_texture_descriptor_set_handle;

        gui->updateGUIElements(frame_info);
        scene_renderer->renderScene(frame_info);
        applyPostProcessing(frame_info);
        gui->renderGUIElements(command_buffer);

        endFrame();
    }
}

VkCommandBuffer Renderer::beginFrame()
{
    assert(!is_frame_started && "Can't call beginFrame while already in progress!");

    auto result = swap_chain->acquireNextImage(&current_image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return nullptr;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    is_frame_started = true;

    auto command_buffer = getCurrentCommandBuffer();
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
    return command_buffer;
}

void Renderer::applyPostProcessing(FrameInfo& frame_info)
{
    beginSwapChainRenderPass(frame_info.command_buffer);
    post_processing->apply(frame_info);
    endSwapChainRenderPass(frame_info.command_buffer);
}

void Renderer::endFrame()
{
    assert(is_frame_started && "Can't call endFrame while frame is not in progress");

    auto command_buffer = getCurrentCommandBuffer();
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer!");
    }

    std::vector<VkCommandBuffer> submit_command_buffers = { command_buffer };
    auto result = swap_chain->submitCommandBuffers(submit_command_buffers.data(), submit_command_buffers.size(), &current_image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.wasWindowResized())
    {
        window.resetWindowResizedFlag();
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to present swap chain image!");
    }

    is_frame_started = false;
    current_frame_index = (current_frame_index + 1) % SwapChain::MAX_FRAMES_IN_FLIGHT;
}

void Renderer::beginSwapChainRenderPass(VkCommandBuffer command_buffer)
{
    assert(is_frame_started && "Can't call beginSwapChainRenderPass if frame is not in progress!");
    assert(command_buffer == getCurrentCommandBuffer() && "Can't begin updateElements pass on command buffer from a different frame!");

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = swap_chain->getRenderPass();
    render_pass_info.framebuffer = swap_chain->getFrameBuffer(current_image_index);

    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain->getSwapChainExtent();

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {0.1f, 0.1f, 0.1f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swap_chain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(swap_chain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, swap_chain->getSwapChainExtent()};
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
}

void Renderer::endSwapChainRenderPass(VkCommandBuffer command_buffer)
{
    assert(is_frame_started && "Can't call endSwapChainRenderPass if frame is not in progress!");
    assert(command_buffer == getCurrentCommandBuffer() && "Can't end updateElements pass on command buffer from a different frame!");

    vkCmdEndRenderPass(command_buffer);
}