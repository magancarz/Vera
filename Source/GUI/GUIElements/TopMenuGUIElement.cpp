#include "TopMenuGUIElement.h"

#include <filesystem>
#include <imgui.h>
#include <imgui_stdlib.h>

#include "editor/editorCommands/SaveProjectCommand.h"

#include "Editor/Editor.h"
#include "editor/editorCommands/ChangeProjectNameCommand.h"
#include "editor/editorCommands/LoadProjectCommand.h"

std::vector<std::shared_ptr<EditorCommand>> TopMenuGUIElement::renderGUIElement(const EditorInfo& editor_info)
{
    std::vector<std::shared_ptr<EditorCommand>> editor_commands;

    ImGui::BeginMainMenuBar();
    if (ImGui::BeginMenu("Project"))
    {
        if (ImGui::MenuItem("Save Project"))
        {
            editor_commands.push_back(std::make_shared<SaveProjectCommand>());
        }

        if (ImGui::MenuItem("Save Project As..."))
        {
            show_saving_project_window = true;
        }

        if (ImGui::MenuItem("Load Project"))
        {
            show_loading_project_window = true;
            selected_project_idx = -1;
            available_project_files.clear();
            for (const auto& entry : std::filesystem::directory_iterator(ProjectUtils::PROJECTS_DIRECTORY))
            {
                std::string file_path = entry.path().generic_string();
                if (file_path.ends_with(ProjectUtils::PROJECT_FILE_EXTENSION))
                {
                    std::string file_name = file_path.substr(ProjectUtils::PROJECTS_DIRECTORY.size());
                    const auto dot_position = file_name.find_first_of(".");
                    file_name = file_name.substr(0, dot_position);
                    available_project_files.push_back(file_name);
                }
            }
        }
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();

    if (show_loading_project_window)
    {
        ImGui::Begin("Load Project");

        for (int i = 0; i < available_project_files.size(); ++i)
        {
            std::string project_name = available_project_files[i];
            if (i == selected_project_idx)
            {
				ImGui::TextColored(ImVec4(1, 1, 0, 1), project_name.c_str());
            }
        	else
            {
                ImGui::Text(project_name.c_str());
            }
            if (ImGui::IsItemClicked())
            {
                selected_project_idx = i;
            }
        }

        if (ImGui::Button("Load"))
        {
            editor_commands.push_back(std::make_shared<LoadProjectCommand>(available_project_files[selected_project_idx >= 0 ? selected_project_idx : 0]));
            show_loading_project_window = false;
        }
        if (ImGui::Button("Exit"))
        {
            show_loading_project_window = false;
        }
        ImGui::End();
    }

    if (show_saving_project_window)
    {
        ImGui::Begin("Save Project As...");

        ImGui::InputText("Project Name", &project_name);

        if (ImGui::Button("Save"))
        {
            editor_commands.push_back(std::make_shared<ChangeProjectNameCommand>(project_name));
            editor_commands.push_back(std::make_shared<SaveProjectCommand>());
            show_saving_project_window = false;
        }
        if (ImGui::Button("Exit"))
        {
            show_saving_project_window = false;
        }
        ImGui::End();
    }

    return editor_commands;
}
