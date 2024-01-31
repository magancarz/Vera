#include "Renderer.h"

#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"

Renderer::Renderer()
{
    loadModels();
    createPipelineLayout();
    createPipeline();
    createCommandBuffers();
}

Renderer::~Renderer()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, VulkanDefines::NO_CALLBACK);
}

void Renderer::renderScene()
{
    uint32_t image_index;
    auto result = swap_chain.acquireNextImage(&image_index);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    result = swap_chain.submitCommandBuffers(&command_buffers[image_index], &image_index);
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to present swap chain image!");
    }
}

void Renderer::loadModels()
{
    std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};
    model = std::make_unique<Model>(device, vertices);
}

void Renderer::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pSetLayouts = nullptr;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device.getDevice(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void Renderer::createPipeline()
{
    auto pipeline_config = Pipeline::defaultPipelineConfigInfo(swap_chain.width(), swap_chain.height());
    pipeline_config.render_pass = swap_chain.getRenderPass();
    pipeline_config.pipeline_layout = pipeline_layout;
    simple_pipeline = std::make_unique<Pipeline>
    (
        device,
        "Shaders/SimpleShader.vert.spv",
        "Shaders/SimpleShader.frag.spv",
        pipeline_config
    );
}

void Renderer::createCommandBuffers()
{
    command_buffers.resize(swap_chain.imageCount());

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool = device.getCommandPool();
    allocate_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

    if (vkAllocateCommandBuffers(device.getDevice(), &allocate_info, command_buffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate command buffers!");
    }

    for (size_t i = 0; i < command_buffers.size(); ++i)
    {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(command_buffers[i], &begin_info) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = swap_chain.getRenderPass();
        render_pass_info.framebuffer = swap_chain.getFrameBuffer(i);

        render_pass_info.renderArea.offset = {0, 0};
        render_pass_info.renderArea.extent = swap_chain.getSwapChainExtent();

        std::array<VkClearValue, 2> clear_values{};
        clear_values[0].color = {0.1f, 0.1f, 0.1f, 1.0f};
        clear_values[1].depthStencil = {1.0f, 0};
        render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();

        vkCmdBeginRenderPass(command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        simple_pipeline->bind(command_buffers[i]);
        model->bind(command_buffers[i]);
        model->draw(command_buffers[i]);

        vkCmdEndRenderPass(command_buffers[i]);
        if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }
}