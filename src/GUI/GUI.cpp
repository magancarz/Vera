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

bool GUI::drawInputFieldForFloat(float* value, const std::string& name, float field_size)
{
    ImGui::Text("%s", name.c_str());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * field_size);
    std::string id = "##" + name;
    ImGui::InputFloat(id.c_str(), value);

    return ImGui::IsItemEdited();
}

std::optional<glm::vec3> GUI::drawInputFieldForVector3(glm::vec3& vector, const std::string& name, float field_size)
{
    float fvector[] =
    {
        vector.x,
        vector.y,
        vector.z
    };
    ImGui::Text("%s", name.c_str());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * field_size);
    std::string id = "##" + name;
    ImGui::InputFloat3(id.c_str(), fvector);

    std::optional<glm::vec3> out;
    if (ImGui::IsItemEdited())
    {
        out = {fvector[0], fvector[1], fvector[2]};
    }
    return out;
}

std::optional<glm::vec4> GUI::drawInputFieldForVector4(glm::vec4& vector, const std::string& name, float field_size)
{
    float fvector[] =
    {
            vector.x,
            vector.y,
            vector.z,
            vector.w
    };
    ImGui::Text("%s", name.c_str());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * field_size);
    std::string id = "##" + name;
    ImGui::InputFloat4(id.c_str(), fvector);

    std::optional<glm::vec4> out;
    if (ImGui::IsItemEdited())
    {
        out = {fvector[0], fvector[1], fvector[2], fvector[3]};
    }
    return out;
}