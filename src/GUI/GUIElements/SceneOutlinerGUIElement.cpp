#include "SceneOutlinerGUIElement.h"

#include <imgui.h>

#include "../../editor/editorCommands/CreateObjectCommand.h"
#include "../../editor/editorCommands/ClearOutlinedObjectsCommand.h"
#include "../../editor/editorCommands/SetObjectToBeOutlinedCommand.h"
#include "editor/Editor.h"
#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"
#include "editor/editorCommands/CreateLightCommand.h"

std::vector<std::shared_ptr<EditorCommand>> SceneOutlinerGUIElement::renderGUIElement(const EditorInfo& editor_info)
{
    std::vector<std::shared_ptr<EditorCommand>> editor_commands;

    ImGui::Begin("Editor", &is_editor_window_visible, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Add"))
        {
            if (ImGui::BeginMenu("Triangle Mesh"))
            {
                for (const auto& model : AssetManager::getAvailableModelAssets())
                {
                    if (ImGui::MenuItem(model->model_name.c_str()))
                    {
                        editor_commands.push_back(std::make_shared<CreateObjectCommand>(model));
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Light"))
            {
                for (const auto& light_creator : AssetManager::getAvailableLightCreators())
                {
                    if (ImGui::MenuItem(light_creator->getLightTypeName().c_str()))
                    {
                        editor_commands.push_back(std::make_shared<CreateLightCommand>(light_creator));
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Scene");
    for (const auto& object : *editor_info.scene_objects)
    {
        ImGui::Text("%s", object->name.c_str());
        if (ImGui::IsItemClicked())
        {
            if (!ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
            {
                editor_commands.push_back(std::make_shared<ClearOutlinedObjectsCommand>());
            }
            editor_commands.push_back(std::make_shared<SetObjectToBeOutlinedCommand>(object));
        }
    }
    ImGui::Separator();

    if (editor_info.outlined_objects->size() == 1)
    {
        editor_info.outlined_objects->at(0)->renderObjectInformationGUI();
    }

    ImGui::End();

    return editor_commands;
}
