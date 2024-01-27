#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../GUI/GUI.h"
#include "../RenderEngine/Renderer.h"

class Editor
{
public:
    Editor();

    void run();

private:
    void renderScene();

    GUI gui_display;
    Renderer master_renderer;
    std::shared_ptr<Camera> camera;
};
