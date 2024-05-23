#pragma once

#include <GLFW/glfw3.h>

#include "InputManager.h"

class GLFWInputManager : public InputManager
{
public:
    explicit GLFWInputManager(GLFWwindow* window);

    GLFWInputManager(const GLFWInputManager&) = delete;
    GLFWInputManager operator=(const GLFWInputManager&) = delete;

    bool isKeyPressed(int key_mapping) override;

private:
    GLFWwindow* window;
};