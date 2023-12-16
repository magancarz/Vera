#include "SceneOutlinerGUIElement.h"

#include <imgui.h>
#include <imgui_stdlib.h>

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
        const auto& outlined_object = (*editor_info.outlined_objects)[0];

        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Object Details");
        ImGui::Text("Name");
        auto& object_name = outlined_object->name;
        ImGui::InputText("##name", &object_name);
        ImGui::Separator();

        const auto object_position = outlined_object->getPosition();
        float position[] =
        {
            object_position.x,
            object_position.y,
            object_position.z
        };
        ImGui::Text("Position");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size * 3);
        ImGui::InputFloat3("##Position", position);
        if (ImGui::IsItemEdited())
        {
            outlined_object->setPosition({position[0], position[1], position[2]});
        }
        const auto object_rotation = outlined_object->getRotation();
        float rotation[] =
        {
            object_rotation.x,
            object_rotation.y,
            object_rotation.z
        };
        ImGui::Text("Rotation");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size * 3);
        ImGui::InputFloat3("##Rotation", rotation);

        if (ImGui::IsItemEdited())
        {
            outlined_object->setRotation({rotation[0], rotation[1], rotation[2]});
        }

        float scale = outlined_object->getScale();
        ImGui::Text("Scale");
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size * 1.33f);
        ImGui::InputFloat("##Scale", &scale);

        if (ImGui::IsItemEdited())
        {
            outlined_object->setScale(scale);
        }

        ImGui::Separator();

        ImGui::Text("Material");
        const auto items = AssetManager::getAvailableMaterialAssets();
        const char* current_item_label = outlined_object->getMaterial()->name.c_str();
        if (ImGui::BeginCombo("##material", current_item_label))
        {
            for (auto& item : items)
            {
                const bool is_selected = (current_item_label == item->material->name);
                if (ImGui::Selectable(item->material->name.c_str(), is_selected))
                {
                    if (ImGui::IsItemEdited())
                    {
                        outlined_object->changeMaterial(item);
                    }

                    current_item_label = item->material->name.c_str();
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::End();

    return editor_commands;
}
