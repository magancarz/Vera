#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"
#include "World/World.h"

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

    World world{window};
    std::unique_ptr<Renderer> renderer;
};