#include "Display.h"

#include <stdexcept>

#include "../input/Input.h"

void Display::keyCallback(GLFWwindow* window, int key, int scan_code, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            Input::set_key_down(key, true);
            break;
        default:
            Input::set_key_down(key, true);
            break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            Input::set_key_down(key, true);
            break;
        default:
            Input::set_key_down(key, false);
            break;
        }
    }
}

void Display::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (button)
        {
        case GLFW_MOUSE_BUTTON_LEFT:
            Input::set_left_mouse_button_down(true);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            Input::set_right_mouse_button_down(true);
            break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (button)
        {
        case GLFW_MOUSE_BUTTON_LEFT:
            Input::set_left_mouse_button_down(false);
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            Input::set_right_mouse_button_down(false);
            break;
        }
    }
}

void Display::cursorPosCallback(GLFWwindow* window, double x_pos, double y_pos)
{
    if (first_mouse)
    {
        last_mouse_x = x_pos;
        last_mouse_y = y_pos;
        first_mouse = false;
    }

    mouse_offset_x = x_pos - last_mouse_x;
    mouse_offset_y = y_pos - last_mouse_y;

    last_mouse_x = x_pos;
    last_mouse_y = y_pos;
}

void Display::scrollCallback(GLFWwindow* window, double x_offset, double y_offset)
{
    mouse_wheel += y_offset;
}

void Display::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    framebuffer_resized = true;
    WINDOW_WIDTH = width;
    WINDOW_HEIGHT = height;
}

void Display::createDisplay()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Couldn't initialize GLFW!\n");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create window!");
    }

    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//    ImGuiIO& io = ImGui::GetIO();
//    (void)io;
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//    ImGui::StyleColorsDark();
//    ImGui_ImplGlfw_InitForOpenGL(window, true);
//    ImGui_ImplOpenGL3_Init("#version 330");

    last_frame_time = getCurrentTime();
}

void Display::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface!");
    }
}

void Display::updateDisplay()
{
    glfwSwapBuffers(window);
    const long current_frame_time = getCurrentTime();
    delta = static_cast<double>(current_frame_time - last_frame_time) / 1000.0;
    last_frame_time = current_frame_time;
}

void Display::closeDisplay()
{
//    ImGui_ImplOpenGL3_Shutdown();
//    ImGui_ImplGlfw_Shutdown();
//    ImGui::DestroyContext();
    glfwTerminate();
}

void Display::checkCloseRequests()
{
    is_close_requested = glfwWindowShouldClose(window);
}

void Display::resetInputValues()
{
    mouse_offset_x = 0;
    mouse_offset_y = 0;
    mouse_wheel = 0;
}

long Display::getCurrentTime()
{
    return static_cast<long>(glfwGetTime() * 1000.0);
}

double Display::getFrameTimeSeconds()
{
    return delta;
}

double Display::getMouseX()
{
    return last_mouse_x;
}

double Display::getMouseY()
{
    return last_mouse_y;
}

void Display::enableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void Display::disableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

double Display::getMouseXOffset()
{
    return mouse_offset_x;
}

double Display::getMouseYOffset()
{
    return mouse_offset_y;
}

double Display::getDWheel()
{
    return mouse_wheel;
}

bool Display::closeNotRequested()
{
    return !is_close_requested;
}
