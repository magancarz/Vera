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

class Vera
{
public:
    Vera() = default;

    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    std::shared_ptr<Window> window = Window::get();
    Device device{*window};

    void loadProject();

    std::shared_ptr<AssetManager> asset_manager = AssetManager::get(&device);
    World world{};

    void createRenderer();

    std::unique_ptr<Renderer> renderer;

    void runLoop();
    void performCleanup();
};