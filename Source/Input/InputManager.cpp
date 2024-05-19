#include "InputManager.h"

#include <cassert>

std::shared_ptr<InputManager> InputManager::get(GLFWwindow* window)
{
    assert(instance || window && "Window can't be nullptr while instance hasn't been created yet!");

    std::lock_guard<std::mutex> lock(mutex);
    if (!instance)
    {
        instance = std::shared_ptr<InputManager>(new InputManager(window));
    }

    return instance;
}

InputManager::InputManager(GLFWwindow* window)
    : window{window} {}

bool InputManager::isKeyPressed(int key_mapping)
{
    return glfwGetKey(window, key_mapping) == GLFW_PRESS;
}