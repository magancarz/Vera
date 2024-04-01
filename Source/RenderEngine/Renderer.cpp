#include "Renderer.h"
#include "GlobalUBO.h"
#include "RenderEngine/Systems/PointLightSystem.h"
#include "RenderEngine/Systems/SimpleRenderSystem.h"

Renderer::Renderer(Window& window, Device& device)
    : window{window}, device{device}
{
    recreateSwapChain();
    createCommandBuffers();
    createDescriptorPool();

    ubo_buffers.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    for (auto& ubo_buffer : ubo_buffers)
    {
        ubo_buffer = std::make_unique<Buffer>
        (
            device,
            sizeof(GlobalUBO),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        );
        ubo_buffer->map();
    }

    auto global_uniform_buffer_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();

    auto global_texture_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();

    global_uniform_buffer_descriptor_sets.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < global_uniform_buffer_descriptor_sets.size(); ++i)
    {
        auto buffer_info = ubo_buffers[i]->descriptorInfo();
        DescriptorWriter(*global_uniform_buffer_set_layout, *global_pool)
                .writeBuffer(0, &buffer_info)
                .build(global_uniform_buffer_descriptor_sets[i]);
    }

    auto first_texture = std::make_shared<Texture>(device, "Resources/Textures/brickwall.png");
    std::vector<std::shared_ptr<Texture>> first_textures = {first_texture};
    auto first_material = std::make_shared<Material>(global_texture_set_layout, global_pool, first_textures);
    materials.push_back(first_material);

    auto second_texture = std::make_shared<Texture>(device, "Resources/Textures/mud.png");
    std::vector<std::shared_ptr<Texture>> second_textures = {second_texture};
    auto second_material = std::make_shared<Material>(global_texture_set_layout, global_pool, second_textures);
    materials.push_back(second_material);

    simple_render_system = std::make_unique<SimpleRenderSystem>
    (
        device,
        getSwapChainRenderPass(),
        global_uniform_buffer_set_layout->getDescriptorSetLayout(),
        global_texture_set_layout->getDescriptorSetLayout()
    );

    point_light_render_system = std::make_unique<PointLightSystem>
    (
        device,
        getSwapChainRenderPass(),
        global_uniform_buffer_set_layout->getDescriptorSetLayout()
    );
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

void Renderer::createDescriptorPool()
{
    global_pool = DescriptorPool::Builder(device)
            .setMaxSets(SwapChain::MAX_FRAMES_IN_FLIGHT * 3)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SwapChain::MAX_FRAMES_IN_FLIGHT)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SwapChain::MAX_FRAMES_IN_FLIGHT * 2)
            .build();
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
        frame_info.frame_index = getFrameIndex();
        frame_info.command_buffer = command_buffer;
        frame_info.global_uniform_buffer_descriptor_set = global_uniform_buffer_descriptor_sets[frame_info.frame_index];

        GlobalUBO ubo{};
        ubo.projection = frame_info.camera->getProjection();
        ubo.view = frame_info.camera->getView();
        ubo.inverse_view = frame_info.camera->getInverseView();
        point_light_render_system->update(frame_info, ubo);
        ubo_buffers[frame_info.frame_index]->writeToBuffer(&ubo);
        ubo_buffers[frame_info.frame_index]->flush();

        beginSwapChainRenderPass(command_buffer);

        simple_render_system->renderObjects(frame_info);
        point_light_render_system->render(frame_info);

        endSwapChainRenderPass(command_buffer);
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

void Renderer::endFrame()
{
    assert(is_frame_started && "Can't call endFrame while frame is not in progress");

    auto command_buffer = getCurrentCommandBuffer();
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer!");
    }

    auto result = swap_chain->submitCommandBuffers(&command_buffer, &current_image_index);
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
    assert(command_buffer == getCurrentCommandBuffer() && "Can't begin render pass on command buffer from a different frame!");

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
    assert(command_buffer == getCurrentCommandBuffer() && "Can't end render pass on command buffer from a different frame!");

    vkCmdEndRenderPass(command_buffer);
}