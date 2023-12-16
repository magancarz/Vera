#include "TinyRayTracingApp.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include "GUI/Display.h"
#include "imgui/imgui_impl_glfw.h"
#include "input/Input.h"

int TinyRayTracingApp::launch()
{
    if (!glfwInit())
        return -1;

    Display::createDisplay();
    Input::initializeInput();

    editor = std::make_shared<Editor>();
    editor->prepare();

    run();
    close();

    return 0;
}

void TinyRayTracingApp::run()
{
    editor->run();
}

void TinyRayTracingApp::close()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    Display::closeDisplay();
}
