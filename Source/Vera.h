#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "RenderEngine/Systems/SimpleRenderSystem.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/Textures/Texture.h"

class Vera
{
public:
    Vera();

    int launch();
    void run();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    void loadObjects();

    Camera camera;

    Window window{1280, 800, "Vera"};
    Device device{window};
    Renderer master_renderer{device, window};
    std::unique_ptr<DescriptorPool> global_pool{};
    std::map<int, Object> objects;

    std::unique_ptr<Texture> texture;
};