#pragma once

#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/Models/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"
#include "World/World.h"
#include "Assets/AssetManager.h"
#include "Utils/VeraDefines.h"
#include "Input/GLFWInputManager.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class Vera
{
public:
    Vera() = default;

    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    void initializeApplication();

    std::shared_ptr<Window> window = Window::get();
    VulkanFacade device{*window};
    std::unique_ptr<MemoryAllocator> memory_allocator;

    std::shared_ptr<AssetManager> asset_manager;
    std::shared_ptr<InputManager> input_manager;

    void loadProject();

    World world{};

    void createRenderer();

    std::unique_ptr<Renderer> renderer;

    void runLoop();
    void performCleanup();
};