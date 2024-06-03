#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include <mutex>

#include "Utils/VeraDefines.h"

class Window {
public:
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const { return glfwWindowShouldClose(window); }
    [[nodiscard]] VkExtent2D getExtent() const { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)}; }
    [[nodiscard]] float getAspect() const { return static_cast<float>(width) / static_cast<float>(height); }
    [[nodiscard]] bool wasWindowResized() const { return framebuffer_resized; }
    void resetWindowResizedFlag() { framebuffer_resized = false; }
    [[nodiscard]] GLFWwindow* getGFLWwindow() const { return window; }

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

    static std::shared_ptr<Window> get(
            uint32_t width = constants::DEFAULT_WINDOW_WIDTH,
            uint32_t height = constants::DEFAULT_WINDOW_HEIGHT,
            std::string name = constants::DEFAULT_WINDOW_TITLE);

private:
    explicit Window(
            uint32_t width = constants::DEFAULT_WINDOW_WIDTH,
            uint32_t height = constants::DEFAULT_WINDOW_HEIGHT,
            std::string name = constants::DEFAULT_WINDOW_TITLE);

    inline static std::shared_ptr<Window> instance;
    inline static std::mutex mutex;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void createWindow();

    uint32_t width;
    uint32_t height;
    bool framebuffer_resized = false;

    std::string window_name;
    GLFWwindow* window;
};