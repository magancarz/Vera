#include "Vera.h"

#include <chrono>

#include "Input/KeyboardMovementController.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/Systems/PointLightSystem.h"

Vera::Vera()
{
    global_pool = DescriptorPool::Builder(device)
            .setMaxSets(SwapChain::MAX_FRAMES_IN_FLIGHT)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SwapChain::MAX_FRAMES_IN_FLIGHT)
            .build();
    loadObjects();
}

int Vera::launch()
{
    run();

    return 0;
}

void Vera::run()
{
    std::vector<std::unique_ptr<Buffer>> ubo_buffers(SwapChain::MAX_FRAMES_IN_FLIGHT);
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

    auto global_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();

    std::vector<VkDescriptorSet> global_descriptor_sets(SwapChain::MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < global_descriptor_sets.size(); ++i)
    {
        auto buffer_info = ubo_buffers[i]->descriptorInfo();
        DescriptorWriter(*global_set_layout, *global_pool)
                .writeBuffer(0, &buffer_info)
                .build(global_descriptor_sets[i]);
    }

    SimpleRenderSystem simple_render_system
    {
        device,
        master_renderer.getSwapChainRenderPass(),
        global_set_layout->getDescriptorSetLayout()
    };

    PointLightSystem point_light_system
    {
        device,
        master_renderer.getSwapChainRenderPass(),
        global_set_layout->getDescriptorSetLayout()
    };

    auto viewer_object = Object::createObject();
    viewer_object.transform_component.translation.z = 2.5f;
    viewer_object.transform_component.rotation.y = glm::radians(180.f);
    KeyboardMovementController movement_controller{};

    camera.setPerspectiveProjection(glm::radians(50.0f), window.getAspect(), 0.1f, 100.f);

    auto current_time = std::chrono::high_resolution_clock::now();
    while (!window.shouldClose())
    {
        glfwPollEvents();

        auto new_time = std::chrono::high_resolution_clock::now();
        float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(new_time - current_time).count();
        current_time = new_time;

        movement_controller.moveInPlaneXZ(window.getGFLWwindow(), frame_time, viewer_object);
        camera.setViewYXZ(viewer_object.transform_component.translation, viewer_object.transform_component.rotation);

        if (auto command_buffer = master_renderer.beginFrame())
        {
            int frame_index = master_renderer.getFrameIndex();
            FrameInfo frame_info{frame_index, frame_time, command_buffer, camera, global_descriptor_sets[frame_index]};

            GlobalUBO ubo{};
            ubo.projection = camera.getProjection();
            ubo.view = camera.getView();
            ubo_buffers[frame_index]->writeToBuffer(&ubo);
            ubo_buffers[frame_index]->flush();

            master_renderer.beginSwapChainRenderPass(command_buffer);

            simple_render_system.renderObjects(frame_info, objects);
            point_light_system.render(frame_info);

            master_renderer.endSwapChainRenderPass(command_buffer);
            master_renderer.endFrame();
        }
    }

    vkDeviceWaitIdle(device.getDevice());
}

void Vera::loadObjects()
{
    std::shared_ptr<Model> monkey_model = Model::createModelFromFile(device, "Resources/Models/monkey.obj");
    auto left_monkey = Object::createObject();
    left_monkey.model = monkey_model;
    left_monkey.color = {0.1f, 0.8f, 0.1f};
    left_monkey.transform_component.translation = {-1.5f, .0f, -2.5f};
    left_monkey.transform_component.rotation = {.0f, .0f, glm::radians(180.0f)};
    left_monkey.transform_component.scale = {.5f, .5f, .5f};
    objects.push_back(std::move(left_monkey));

    auto right_monkey = Object::createObject();
    right_monkey.model = monkey_model;
    right_monkey.color = {1.f, 1.f, 1.f};
    right_monkey.transform_component.translation = {1.5f, .0f, -2.5f};
    right_monkey.transform_component.rotation = {.0f, 0.f, glm::radians(180.0f)};
    right_monkey.transform_component.scale = {.5f, .5f, .5f};
    objects.push_back(std::move(right_monkey));

    std::shared_ptr<Model> plane_model = Model::createModelFromFile(device, "Resources/Models/plane.obj");
    auto plane = Object::createObject();
    plane.model = plane_model;
    plane.color = {0.1f, 0.8f, 0.1f};
    plane.transform_component.translation = {0.f, 1.f, -2.5f};
    plane.transform_component.rotation = {.0f, 0.f, glm::radians(180.0f)};
    plane.transform_component.scale = {3.f, 3.f, 3.f};
    objects.push_back(std::move(plane));
}
