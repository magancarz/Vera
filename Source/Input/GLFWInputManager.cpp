#include "GLFWInputManager.h"

GLFWInputManager::GLFWInputManager(GLFWwindow* window)
        : window{window} {}

bool GLFWInputManager::isKeyPressed(int key_mapping)
{
    return glfwGetKey(window, key_mapping) == GLFW_PRESS;
}