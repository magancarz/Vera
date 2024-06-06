#pragma once

#include "RenderEngine/Renderer.h"
#include "World/World.h"
#include "Assets/AssetManager.h"
#include "Input/GLFWInputManager.h"
#include "Memory/MemoryAllocator.h"

class Vera
{
public:
    Vera();
    ~Vera();

    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    void initializeAppComponents();
    void initializeLogSystem();

    Window* window{nullptr};
    std::unique_ptr<InputManager> input_manager{nullptr};

    std::unique_ptr<VulkanFacade> vulkan_facade{nullptr};

    std::unique_ptr<MemoryAllocator> memory_allocator{nullptr};
    std::unique_ptr<AssetManager> asset_manager{nullptr};

    void loadProject();

    World world{};

    void createRenderer();

    std::unique_ptr<Renderer> renderer;
};