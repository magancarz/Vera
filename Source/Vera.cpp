#include "Vera.h"

#include <chrono>

#include "Input/KeyboardMovementController.h"

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
            master_renderer.beginSwapChainRenderPass(command_buffer);
            simple_render_system.renderObjects(command_buffer, objects, camera);
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
    auto model = createCubeModel(device, {0, 0, 0});
    auto cube = Object::createObject();
    cube.model = std::move(model);
    cube.color = {0.1f, 0.8f, 0.1f};
    cube.transform_component.translation = {.0f, .0f, -2.5f};
    cube.transform_component.scale = {.5f, .5f, .5f};

    objects.push_back(std::move(cube));
}

// temporary helper function, creates a 1x1x1 cube centered at offset
std::unique_ptr<Model> Vera::createCubeModel(Device& device, glm::vec3 offset) {
    Model::Builder builder{};

    builder.vertices =
    {
            // left face (white)
            {{-.5f, -.5f, -.5f}, {.9f, .9f, .9f}},
            {{-.5f, .5f, .5f}, {.9f, .9f, .9f}},
            {{-.5f, -.5f, .5f}, {.9f, .9f, .9f}},
            {{-.5f, .5f, -.5f}, {.9f, .9f, .9f}},

            // right face (yellow)
            {{.5f, -.5f, -.5f}, {.8f, .8f, .1f}},
            {{.5f, .5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, -.5f, .5f}, {.8f, .8f, .1f}},
            {{.5f, .5f, -.5f}, {.8f, .8f, .1f}},

            // top face (orange, remember y axis points down)
            {{-.5f, -.5f, -.5f}, {.9f, .6f, .1f}},
            {{.5f, -.5f, .5f}, {.9f, .6f, .1f}},
            {{-.5f, -.5f, .5f}, {.9f, .6f, .1f}},
            {{.5f, -.5f, -.5f}, {.9f, .6f, .1f}},

            // bottom face (red)
            {{-.5f, .5f, -.5f}, {.8f, .1f, .1f}},
            {{.5f, .5f, .5f}, {.8f, .1f, .1f}},
            {{-.5f, .5f, .5f}, {.8f, .1f, .1f}},
            {{.5f, .5f, -.5f}, {.8f, .1f, .1f}},

            // nose face (blue)
            {{-.5f, -.5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f, .5f, 0.5f}, {.1f, .1f, .8f}},
            {{-.5f, .5f, 0.5f}, {.1f, .1f, .8f}},
            {{.5f, -.5f, 0.5f}, {.1f, .1f, .8f}},

            // tail face (green)
            {{-.5f, -.5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f, .5f, -0.5f}, {.1f, .8f, .1f}},
            {{-.5f, .5f, -0.5f}, {.1f, .8f, .1f}},
            {{.5f, -.5f, -0.5f}, {.1f, .8f, .1f}},

    };
    for (auto& v : builder.vertices)
    {
        v.position += offset;
    }

    builder.indices = {0,  1,  2,  0,  3,  1,  4,  5,  6,  4,  7,  5,  8,  9,  10, 8,  11, 9,
                            12, 13, 14, 12, 15, 13, 16, 17, 18, 16, 19, 17, 20, 21, 22, 20, 23, 21};

    return std::make_unique<Model>(device, builder);
}
