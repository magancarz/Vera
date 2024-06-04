#pragma once

#include <vulkan/vulkan_core.h>
#include "GLFW/glfw3.h"

#include "Utils/VeraDefines.h"
#include "Window.h"

class GLFWWindow : public Window
{
public:
    explicit GLFWWindow(
            uint32_t width = constants::DEFAULT_WINDOW_WIDTH,
            uint32_t height = constants::DEFAULT_WINDOW_HEIGHT,
            std::string name = constants::DEFAULT_WINDOW_TITLE);
    ~GLFWWindow() override;

    GLFWWindow(const GLFWWindow&) = delete;
    GLFWWindow& operator=(const GLFWWindow&) = delete;

    [[nodiscard]] bool shouldClose() const override { return glfwWindowShouldClose(window); }
    [[nodiscard]] GLFWwindow* getGFLWwindow() const { return window; }

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void createWindow();

    GLFWwindow* window{nullptr};
};