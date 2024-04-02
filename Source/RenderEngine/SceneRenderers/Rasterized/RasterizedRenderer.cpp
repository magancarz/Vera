#include "RasterizedRenderer.h"

RasterizedRenderer::RasterizedRenderer(Device& device, VkRenderPass render_pass)
    : device{device}
{
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
        render_pass,
        global_uniform_buffer_set_layout->getDescriptorSetLayout(),
        global_texture_set_layout->getDescriptorSetLayout()
    );

    point_light_render_system = std::make_unique<PointLightSystem>
    (
        device,
        render_pass,
        global_uniform_buffer_set_layout->getDescriptorSetLayout()
    );
}

void RasterizedRenderer::createDescriptorPool()
{
    global_pool = DescriptorPool::Builder(device)
            .setMaxSets(SwapChain::MAX_FRAMES_IN_FLIGHT * 3)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SwapChain::MAX_FRAMES_IN_FLIGHT)
            .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SwapChain::MAX_FRAMES_IN_FLIGHT * 2)
            .build();
}

void RasterizedRenderer::renderScene(FrameInfo& frame_info)
{
    frame_info.global_uniform_buffer_descriptor_set = global_uniform_buffer_descriptor_sets[frame_info.frame_index];

    GlobalUBO ubo{};
    ubo.projection = frame_info.camera->getProjection();
    ubo.view = frame_info.camera->getView();
    ubo.inverse_view = frame_info.camera->getInverseView();
    point_light_render_system->update(frame_info, ubo);
    ubo_buffers[frame_info.frame_index]->writeToBuffer(&ubo);
    ubo_buffers[frame_info.frame_index]->flush();

    simple_render_system->renderObjects(frame_info);
    point_light_render_system->render(frame_info);
}