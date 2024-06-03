#include "Vera.h"

#include <chrono>

#include "RenderEngine/SceneRenderers/RayTraced/RayTracedRenderer.h"
#include "RenderEngine/Memory/Vulkan/VulkanMemoryAllocator.h"

Vera::Vera()
{
    initializeApplication();
    loadProject();
    createRenderer();
}

void Vera::initializeApplication()
{
    memory_allocator = std::make_unique<VulkanMemoryAllocator>(vulkan_facade);
    asset_manager = std::make_unique<AssetManager>(vulkan_facade, *memory_allocator);

    input_manager = std::make_unique<GLFWInputManager>(window->getGFLWwindow());
}

void Vera::loadProject()
{
    ProjectInfo project_info = ProjectUtils::loadProject("vera");
    world.loadProject(project_info, *asset_manager);
    world.createViewerObject(*input_manager);
}

void Vera::createRenderer()
{
    renderer = std::make_unique<Renderer>(*window, vulkan_facade, *memory_allocator, world, *asset_manager);
}

Vera::~Vera()
{
    performCleanup();
}

void Vera::performCleanup()
{
    asset_manager->clearResources();
    vkDeviceWaitIdle(vulkan_facade.getDevice());
}

void Vera::run()
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