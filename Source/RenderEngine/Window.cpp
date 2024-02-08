#include "Window.h"

#include <stdexcept>

Window::Window(int width, int height, std::string name)
    : width{width}, height{height}, window_name{std::move(name)}
{
    initWindow();
}

Window::~Window()
{
//    ImGui_ImplOpenGL3_Shutdown();
//    ImGui_ImplGlfw_Shutdown();
//    ImGui::DestroyContext();
    glfwTerminate();
}

void Window::initWindow()
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

//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//    ImGuiIO& io = ImGui::GetIO();
//    (void)io;
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//    ImGui::StyleColorsDark();
//    ImGui_ImplGlfw_InitForOpenGL(window, true);
//    ImGui_ImplOpenGL3_Init("#version 330");
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
