#include "Editor.h"

#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

#include "GUI/Display.h"
#include "RenderEngine/Camera.h"
#include "Utils/Timer.h"
#include "Input/Input.h"

Editor::Editor()
{
    camera = std::make_shared<Camera>(glm::vec3(0, 10, 7));
}

void Editor::run()
{
    while (Display::closeNotRequested())
    {
        glfwPollEvents();

        Display::resetInputValues();

        renderScene();

        Display::updateDisplay();

        Display::checkCloseRequests();
    }
}

void Editor::renderScene()
{
    master_renderer.renderScene();
}
