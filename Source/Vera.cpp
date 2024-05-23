#include "Vera.h"

#include <chrono>

#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/SceneRenderers/RayTraced/RayTracedRenderer.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

void Vera::run()
{
    initializeApplication();
    loadProject();
    createRenderer();
    runLoop();
    performCleanup();
}

void Vera::initializeApplication()
{
    asset_manager = std::make_shared<AssetManager>(device);
    input_manager = std::make_shared<GLFWInputManager>(window->getGFLWwindow());
}
void Vera::loadProject()
{
    ProjectInfo project_info = ProjectUtils::loadProject("vera");
    world.loadProject(project_info, asset_manager);
    world.createViewerObject(input_manager);
}

void Vera::createRenderer()
{
    renderer = std::make_unique<Renderer>(*window, device, world, asset_manager);
}

void Vera::runLoop()
{
    auto last_time = std::chrono::high_resolution_clock::now();
    while (!window->shouldClose())
    {
        glfwPollEvents();

        auto current_time = std::chrono::high_resolution_clock::now();
        float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();
        last_time = current_time;

        FrameInfo frame_info{};
        frame_info.frame_time = frame_time;
        world.update(frame_info);
        renderer->render(frame_info);
    }
}

void Vera::performCleanup()
{
    asset_manager->clearResources();
    vkDeviceWaitIdle(device.getDevice());
}