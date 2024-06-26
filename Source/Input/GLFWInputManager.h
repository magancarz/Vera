#pragma once

#include <GLFW/glfw3.h>

#include "InputManager.h"

class GLFWInputManager : public InputManager
{
public:
    GLFWInputManager();

    bool isKeyPressed(KeyCode key_mapping) override;

private:
    void obtainGLFWwindowPointer();

    GLFWwindow* window{nullptr};
};