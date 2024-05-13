#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/Models/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"
#include "World/World.h"
#include "Assets/AssetManager.h"

class Vera
{
public:
    Vera() = default;

    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    Window window{1280, 800, "Vera"};
    Device device{window};

    std::shared_ptr<AssetManager> asset_manager = AssetManager::get(&device);
    World world{window};
    std::unique_ptr<Renderer> renderer;
};