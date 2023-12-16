#include "GUI.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "../editor/Editor.h"
#include "GUIElements/RayTracerGUIElement.h"
#include "GUIElements/SceneOutlinerGUIElement.h"
#include "GUIElements/TopMenuGUIElement.h"

GUI::GUI()
{
    gui_elements.push_back(std::make_unique<TopMenuGUIElement>());
    gui_elements.push_back(std::make_unique<RayTracerGUIElement>());
    gui_elements.push_back(std::make_unique<SceneOutlinerGUIElement>());
}

std::vector<std::shared_ptr<EditorCommand>> GUI::renderGUI(const EditorInfo& editor_info)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::vector<std::shared_ptr<EditorCommand>> editor_requests;
    for (const auto& gui_element : gui_elements)
    {
        auto editor_commands = gui_element->renderGUIElement(editor_info);
        for (const auto& command : editor_commands)
        {
            editor_requests.push_back(command);
        }
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return editor_requests;
}
