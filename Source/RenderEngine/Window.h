#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>

class Window {
public:
    Window(int width, int height, std::string name);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() { return glfwWindowShouldClose(window); }
    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
    bool wasWindowResized() { return framebuffer_resized; }
    void resetWindowResizedFlag() { framebuffer_resized = false; }

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void initWindow();

    int width;
    int height;
    bool framebuffer_resized = false;

    std::string window_name;
    GLFWwindow* window;
};