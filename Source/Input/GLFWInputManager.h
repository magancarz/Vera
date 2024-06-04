#pragma once

#include <GLFW/glfw3.h>

#include "InputManager.h"

class GLFWInputManager : public InputManager
{
public:
    GLFWInputManager();

    GLFWInputManager(const GLFWInputManager&) = delete;
    GLFWInputManager operator=(const GLFWInputManager&) = delete;

    bool isKeyPressed(KeyCode key_mapping) override;

private:
    void obtainGLFWwindowPointer();

    GLFWwindow* window{nullptr};
};