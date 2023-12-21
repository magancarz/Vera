#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../RenderEngine/RayTracing//RayTracerConfig.h"

class Display
{
public:
    static void createDisplay();
    static void updateDisplay();
    static void closeDisplay();

    static void checkCloseRequests();

    static void keyCallback(GLFWwindow* window, int key, int scan_code, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double x_pos, double y_pos);
    static void scrollCallback(GLFWwindow* window, double x_offset, double y_offset);

    static void resetInputValues();

    static double getMouseX();
    static double getMouseY();
    static void enableCursor();
    static void disableCursor();
    static double getMouseXOffset();
    static double getMouseYOffset();
    static double getDWheel();

    static long getCurrentTime();
    static double getFrameTimeSeconds();
    static bool closeNotRequested();

    inline static const int WINDOW_WIDTH = 1280;
    inline static const int WINDOW_HEIGHT = 800;

    inline static const char* WINDOW_TITLE = "Ray Tracing";

private:
    inline static bool is_close_requested = false;
    inline static bool is_input_enabled = true;
    inline static GLFWwindow* window;

    inline static long last_frame_time;
    inline static double delta;

    inline static double mouse_offset_x = 0, mouse_offset_y = 0;
    inline static double last_mouse_x = 0, last_mouse_y = 0;
    inline static bool first_mouse = true;

    inline static double mouse_wheel = 0;
};
