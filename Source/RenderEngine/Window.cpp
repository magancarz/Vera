#include "Window.h"

#include <stdexcept>

std::shared_ptr<Window> Window::get(uint32_t width, uint32_t height, std::string name)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!instance)
    {
        instance = std::shared_ptr<Window>(new Window(width, height, std::move(name)));
    }

    return instance;
}

Window::Window(uint32_t width, uint32_t height, std::string name)
    : width{width}, height{height}, window_name{std::move(name)}
{
    createWindow();
}

Window::~Window()
{
    glfwTerminate();
}

void Window::createWindow()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Couldn't initialize GLFW!\n");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(width, height, window_name.c_str(), nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create window!");
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void Window::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface");
    }
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    window_ptr->framebuffer_resized = true;
    window_ptr->width = width;
    window_ptr->height = height;
}
