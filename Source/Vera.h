#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "GUI/Display.h"
#include "RenderEngine/RenderingAPI/Model.h"
#include "Objects/Object.h"
#include "RenderEngine/Renderer.h"
#include "GUI/GUI.h"
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

    GUI gui_display;
    std::shared_ptr<Camera> camera;

    Device device;
    Renderer master_renderer{device};
    SimpleRenderSystem simple_render_system{device, master_renderer.getSwapChainRenderPass()};
    std::vector<Object> objects;
};