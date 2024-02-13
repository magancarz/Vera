#include "Vera.h"

#include <chrono>

#include "Input/KeyboardMovementController.h"
#include "RenderEngine/GlobalUBO.h"

int Vera::launch()
{
    run();
    close();

    return 0;
}

void Vera::run()
{
    loadObjects();

    auto viewer_object = Object::createObject();
    KeyboardMovementController movement_controller{};

    Buffer global_ubo_buffer
    {
        device,
        sizeof(GlobalUBO),
        SwapChain::MAX_FRAMES_IN_FLIGHT,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        device.properties.limits.minUniformBufferOffsetAlignment
    };
    global_ubo_buffer.map();

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
            FrameInfo frame_info{frame_index, frame_time, command_buffer, camera};

//            GlobalUBO ubo{};
//            ubo.projection_view = camera.getPerspectiveProjectionMatrix(window.getAspect()) * camera.getViewMatrix();
//            global_ubo_buffer.writeToIndex(&ubo, frame_index);
//            global_ubo_buffer.flushIndex(frame_index);

            master_renderer.beginSwapChainRenderPass(command_buffer);
            simple_render_system.renderObjects(frame_info, objects);
            master_renderer.endSwapChainRenderPass(command_buffer);
            master_renderer.endFrame();
        }
    }
}

void Vera::close()
{
    vkDeviceWaitIdle(device.getDevice());
}

void Vera::loadObjects()
{
    auto model = Model::createModelFromFile(device, "Resources/Models/monkey.obj");
    auto cube = Object::createObject();
    cube.model = std::move(model);
    cube.color = {0.1f, 0.8f, 0.1f};
    cube.transform_component.translation = {.0f, .0f, -2.5f};
    cube.transform_component.rotation = {.0f, .0f, glm::radians(180.0f)};
    cube.transform_component.scale = {.5f, .5f, .5f};

    objects.push_back(std::move(cube));
}
