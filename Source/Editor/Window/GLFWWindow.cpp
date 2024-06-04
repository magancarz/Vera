#include "GLFWWindow.h"

#include <stdexcept>

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, std::string name)
    : Window{width, height, std::move(name)}
{
    createWindow();
}

GLFWWindow::~GLFWWindow()
{
    glfwTerminate();
}

void GLFWWindow::createWindow()
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

void GLFWWindow::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface");
    }
}

void GLFWWindow::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto window_ptr = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    window_ptr->window_resized = true;
    window_ptr->width = width;
    window_ptr->height = height;
}