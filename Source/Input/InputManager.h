#pragma once

#include <GLFW/glfw3.h>

#include <memory>
#include <mutex>

class InputManager
{
public:
    InputManager(const InputManager&) = delete;
    InputManager operator=(const InputManager&) = delete;

    ~InputManager() = default;

    bool isKeyPressed(int key_mapping);

    static std::shared_ptr<InputManager> get(GLFWwindow* window = nullptr);

private:
    explicit InputManager(GLFWwindow* window);

    inline static std::shared_ptr<InputManager> instance;
    inline static std::mutex mutex;

    GLFWwindow* window;
};