#include "Vera.h"

#include "GUI/Display.h"

int Vera::launch()
{
    run();
    close();

    return 0;
}

void Vera::run()
{
    loadObjects();

    while (Display::closeNotRequested())
    {
        glfwPollEvents();
        Display::resetInputValues();

        if (auto command_buffer = master_renderer.beginFrame())
        {
            master_renderer.beginSwapChainRenderPass(command_buffer);
            simple_render_system.renderObjects(command_buffer, objects);
            master_renderer.endSwapChainRenderPass(command_buffer);
            master_renderer.endFrame();
        }

        Display::updateDisplay();
        Display::checkCloseRequests();
    }
}

void Vera::close()
{
    vkDeviceWaitIdle(device.getDevice());
    Display::closeDisplay();
}

void Vera::loadObjects()
{
    std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};
    auto model = std::make_shared<Model>(device, vertices);

    auto triangle = Object::createObject();
    triangle.model = model;
    triangle.color = {0.1f, 0.8f, 0.1f};
    triangle.transform_2d.translation.x = 0.f;

    objects.push_back(std::move(triangle));
}
