#include "Vera.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include "GUI/Display.h"
#include "imgui/imgui_impl_glfw.h"
#include "input/Input.h"

int Vera::launch()
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

void Vera::run()
{
    editor->run();
}

void Vera::close()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    Display::closeDisplay();
}
