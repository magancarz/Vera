#include "Vera.h"

#include <chrono>

#include "Editor/Window/GLFWWindow.h"
#include "Editor/Window/WindowSystem.h"
#include "Logs/BasicLogger.h"
#include "Logs/FileLogger.h"
#include "Logs/LogSystem.h"
#include "Memory/Vulkan/VulkanMemoryAllocator.h"
#include "Logs/Defines.h"
#include "Utils/PathBuilder.h"

Vera::Vera()
{
    initializeAppComponents();
    loadProject();
    createRenderer();
}

void Vera::initializeAppComponents()
{
    initializeLogSystem();

    window = WindowSystem::initialize(std::make_unique<GLFWWindow>());
    input_manager = std::make_unique<GLFWInputManager>();

    vulkan_facade = std::make_unique<VulkanHandler>();
    memory_allocator = std::make_unique<VulkanMemoryAllocator>(*vulkan_facade);
    asset_manager = std::make_unique<AssetManager>(*vulkan_facade, *memory_allocator);
}

void Vera::initializeLogSystem()
{
    auto basic_logger = std::make_unique<BasicLogger>();
    std::string log_file_location = PathBuilder().append(Logs::LOG_FOLDER_LOCATION).append(Logs::DEFAULT_LOG_FILE_NAME).build();
    auto file_logger_decorator = std::make_unique<FileLogger>(std::move(basic_logger), log_file_location);
    LogSystem::initialize(std::move(file_logger_decorator));
}

void Vera::loadProject()
{
    ProjectInfo project_info = ProjectUtils::loadProject("vera");
    world.loadProject(project_info, *asset_manager);
    world.createViewerObject(*input_manager);
}

void Vera::createRenderer()
{
    renderer = std::make_unique<Renderer>(*window, *vulkan_facade, *memory_allocator, world, *asset_manager);
}

Vera::~Vera()
{
    vkDeviceWaitIdle(vulkan_facade->getDeviceHandle());
}

void Vera::run()
{
    auto last_time = std::chrono::high_resolution_clock::now();
    while (!window->shouldClose())
    {
        glfwPollEvents();

        FrameInfo frame_info{};

        auto current_time = std::chrono::high_resolution_clock::now();
        frame_info.delta_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();
        last_time = current_time;

        world.update(frame_info);
        renderer->render(frame_info);
    }
}