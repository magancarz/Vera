#pragma once

#include "RenderEngine/Renderer.h"
#include "World/World.h"
#include "Assets/AssetManager.h"
#include "Input/GLFWInputManager.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class Vera
{
public:
    Vera();
    ~Vera();

    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    void initializeApplication();

    std::shared_ptr<Window> window = Window::get();
    VulkanFacade vulkan_facade{*window};
    std::unique_ptr<MemoryAllocator> memory_allocator;

    std::unique_ptr<AssetManager> asset_manager;
    std::unique_ptr<InputManager> input_manager;

    void loadProject();

    World world{};

    void createRenderer();

    std::unique_ptr<Renderer> renderer;

    void runLoop();
    void performCleanup();
};