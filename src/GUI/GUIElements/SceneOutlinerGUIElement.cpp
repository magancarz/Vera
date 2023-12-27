#include "SceneOutlinerGUIElement.h"

#include <imgui.h>

#include "../../editor/editorCommands/CreateObjectCommand.h"
#include "../../editor/editorCommands/ClearOutlinedObjectsCommand.h"
#include "../../editor/editorCommands/SetObjectToBeOutlinedCommand.h"
#include "editor/Editor.h"
#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"

std::vector<std::shared_ptr<EditorCommand>> SceneOutlinerGUIElement::renderGUIElement(const EditorInfo& editor_info)
{
    std::vector<std::shared_ptr<EditorCommand>> editor_commands;

    ImGui::Begin("Editor", &is_editor_window_visible, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Add"))
        {
            if (ImGui::MenuItem("Sphere"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("sphere")));
            }
            if (ImGui::MenuItem("Icosphere"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("icosphere")));
            }
            if (ImGui::MenuItem("Cube"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("cube")));
            }
            if (ImGui::MenuItem("Plane"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("plane")));
            }
            if (ImGui::MenuItem("Barrel"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("barrel")));
            }
            if (ImGui::MenuItem("Cherry"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("cherry")));
            }
            if (ImGui::MenuItem("Lantern"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("lantern")));
            }
            if (ImGui::MenuItem("Rock"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("rock")));
            }
            if (ImGui::MenuItem("Monkey"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("monkey")));
            }
            if (ImGui::MenuItem("Teapot"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("teapot")));
            }
            if (ImGui::MenuItem("Bunny"))
            {
                editor_commands.push_back(std::make_shared<CreateObjectCommand>(AssetManager::findModelAsset("bunny")));
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
