#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include "Utils/VeraDefines.h"

class Window {
public:
    Window(
            uint32_t width = constants::DEFAULT_WINDOW_WIDTH,
            uint32_t height = constants::DEFAULT_WINDOW_HEIGHT,
            std::string name = constants::DEFAULT_WINDOW_TITLE);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() { return glfwWindowShouldClose(window); }
    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
    float getAspect() { return static_cast<float>(width) / static_cast<float>(height); }
    bool wasWindowResized() { return framebuffer_resized; }
    void resetWindowResizedFlag() { framebuffer_resized = false; }
    GLFWwindow* getGFLWwindow() const { return window; }

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void createWindow();

    uint32_t width;
    uint32_t height;
    bool framebuffer_resized = false;

    std::string window_name;
    GLFWwindow* window;
};