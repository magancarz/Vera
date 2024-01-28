#pragma once

#include "Camera.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "GUI/Display.h"

class Renderer
{
public:
    Renderer();

    void renderScene();

private:
    Device device;
    Pipeline simple_pipeline
    {
        device,
        "Shaders/SimpleShader.vert.spv",
        "Shaders/SimpleShader.frag.spv",
        Pipeline::defaultPipelineConfigInfo(Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT)
    };
};
