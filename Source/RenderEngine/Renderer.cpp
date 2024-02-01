#include "Renderer.h"

#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"

Renderer::Renderer()
{
    loadObjects();
    createPipelineLayout();
    recreateSwapChain();
    createCommandBuffers();
}

Renderer::~Renderer()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, VulkanDefines::NO_CALLBACK);
}

void Renderer::renderScene()
{
    uint32_t image_index;
    auto result = swap_chain->acquireNextImage(&image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    recordCommandBuffer(image_index);
    result = swap_chain->submitCommandBuffers(&command_buffers[image_index], &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || Display::wasWindowResized())
    {
        Display::resetWindowResizedFlag();
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to present swap chain image!");
    }
}

void Renderer::loadObjects()
{
    std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};
    auto model = std::make_shared<Model>(device, vertices);

    auto triangle = Object::createObject();
    triangle.model = model;
    triangle.color = {0.1f, 0.8f, 0.1f};
    triangle.transform_2d.translation.x = .2f;

    objects.push_back(std::move(triangle));
}

void Renderer::createPipelineLayout()
{
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(SimplePushConstantData);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pSetLayouts = nullptr;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(device.getDevice(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void Renderer::createPipeline()
{
    assert(swap_chain != nullptr && "Cannot create pipeline before swap chain");
    assert(pipeline_layout != nullptr && "Cannot create pipeline before pipeline layout");

    PipelineConfigInfo config_info{};
    Pipeline::defaultPipelineConfigInfo(config_info);
    config_info.render_pass = swap_chain->getRenderPass();
    config_info.pipeline_layout = pipeline_layout;
    simple_pipeline = std::make_unique<Pipeline>
    (
        device,
        "Shaders/SimpleShader.vert.spv",
        "Shaders/SimpleShader.frag.spv",
        config_info
    );
}

void Renderer::createCommandBuffers()
{
    command_buffers.resize(swap_chain->imageCount());

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

void Renderer::freeCommandBuffers()
{
    vkFreeCommandBuffers(
            device.getDevice(),
            device.getCommandPool(),
            static_cast<uint32_t>(command_buffers.size()),
            command_buffers.data());
    command_buffers.clear();
}

void Renderer::recreateSwapChain()
{
    auto extent = Display::getExtent();
    while (extent.width == 0 || extent.height == 0)
    {
        extent = Display::getExtent();
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device.getDevice());
    swap_chain.reset();
    swap_chain = std::make_unique<SwapChain>(device, extent);
    if (swap_chain == nullptr)
    {
        swap_chain = std::make_unique<SwapChain>(device, extent);
    }
    else
    {
        std::shared_ptr<SwapChain> old_swap_chain = std::move(swap_chain);
        swap_chain = std::make_unique<SwapChain>(device, extent, std::move(old_swap_chain));
        if (swap_chain->imageCount() != command_buffers.size())
        {
            freeCommandBuffers();
            createCommandBuffers();
        }
    }
    createPipeline();
}

void Renderer::recordCommandBuffer(int image_index)
{
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(command_buffers[image_index], &begin_info) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = swap_chain->getRenderPass();
    render_pass_info.framebuffer = swap_chain->getFrameBuffer(image_index);

    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain->getSwapChainExtent();

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {0.1f, 0.1f, 0.1f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffers[image_index], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swap_chain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(swap_chain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, swap_chain->getSwapChainExtent()};
    vkCmdSetViewport(command_buffers[image_index], 0, 1, &viewport);
    vkCmdSetScissor(command_buffers[image_index], 0, 1, &scissor);

    renderObjects(command_buffers[image_index]);

    vkCmdEndRenderPass(command_buffers[image_index]);
    if (vkEndCommandBuffer(command_buffers[image_index]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

void Renderer::renderObjects(VkCommandBuffer command_buffer)
{
    simple_pipeline->bind(command_buffer);

    for (auto& obj : objects)
    {
        SimplePushConstantData push{};
        push.offset = obj.transform_2d.translation;
        push.color = obj.color;
        push.transform = obj.transform_2d.mat2();

        vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SimplePushConstantData),
                &push
        );
        obj.model->bind(command_buffer);
        obj.model->draw(command_buffer);
    }
}