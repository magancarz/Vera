#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "RenderEngine/SimpleRenderSystem.h"

class Vera
{
public:
    Vera() = default;

    int launch();
    void run();
    void close();

    Vera(const Vera&) = delete;
    Vera& operator=(const Vera&) = delete;

private:
    void loadObjects();
    std::unique_ptr<Model> createCubeModel(Device& device, glm::vec3 offset);

    Camera camera{{0, 0, 5}};

    Window window{1280, 800, "Vera"};
    Device device{window};
    Renderer master_renderer{device, window};
    SimpleRenderSystem simple_render_system{device, master_renderer.getSwapChainRenderPass()};
    std::vector<Object> objects;
};