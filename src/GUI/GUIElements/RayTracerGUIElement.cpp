#include "RayTracerGUIElement.h"

#include <imgui.h>
#include <imgui_stdlib.h>
#include <iomanip>
#include <sstream>

#include "../../editor/editorCommands/RefreshRayTracerConfigCommand.h"
#include "../../editor/editorCommands/GenerateRayTracedImagesQueueCommand.h"
#include "../../editor/editorCommands/ToggleLiveRayTracingCommand.h"
#include "Editor/EditorCommands/StopGeneratingImageCommand.h"
#include "Editor/Editor.h"

std::vector<std::shared_ptr<EditorCommand>> RayTracerGUIElement::renderGUIElement(const EditorInfo& editor_info)
{
    std::vector<std::shared_ptr<EditorCommand>> editor_commands;

    ImGui::Begin("Ray Tracing Settings");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar("Ray Traced Image Width", ImGuiDataType_U32, &image_width);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar("Ray Traced Image Height", ImGuiDataType_U32, &image_height);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar("Number Of Ray Bounces", ImGuiDataType_U32, &number_of_ray_bounces);
    if (ImGui::Button("Refresh Ray Tracer Config"))
    {
        RayTracerConfig config{
            image_width,
            image_height,
            number_of_ray_bounces
        };
        editor_commands.push_back(std::make_shared<RefreshRayTracerConfigCommand>(config));
    }
    if (ImGui::Button("Live Ray Tracing"))
    {
        editor_commands.push_back(std::make_shared<ToggleLiveRayTracingCommand>());
    }

    ImGui::Separator();
    
	renderImageQueue(editor_info, editor_commands);

	ImGui::End();

    return editor_commands;
}

void RayTracerGUIElement::renderImageQueue(const EditorInfo& editor_info, std::vector<std::shared_ptr<EditorCommand>>& editor_commands)
{
    if (ImGui::Button("Generate Images From Queue"))
    {
        editor_commands.push_back(std::make_shared<GenerateRayTracedImagesQueueCommand>(&ray_traced_images_queue));
    }

    for (int i = 0; i < ray_traced_images_queue.size(); ++i)
	{
        auto& image_config = ray_traced_images_queue[i];
        if (editor_info.generating_image && i == 0)
        {
            renderCurrentlyGeneratedImageQueueElement(editor_info, editor_commands, image_config);
            continue;
        }
        renderImageQueueElement(image_config);
	}

    if (!editor_info.generating_image)
    {
        if (ImGui::Button("Add Element"))
        {
            RayTracerConfig new_image_config{};
            const auto time_of_creation = std::chrono::system_clock::now();
            new_image_config.image_name += "_" + std::to_string(time_of_creation.time_since_epoch().count());
            ray_traced_images_queue.push_back(new_image_config);
        }

        ImGui::SameLine();

        if (ImGui::Button("Remove Last Element"))
        {
            ray_traced_images_queue.pop_back();
        }
    }
}

void RayTracerGUIElement::renderCurrentlyGeneratedImageQueueElement(const EditorInfo& editor_info, std::vector<std::shared_ptr<EditorCommand>>& editor_commands, RayTracerConfig& image_config)
{
    const std::string image_name = std::string("Image Name: " + image_config.image_name);
    const std::string image_width = std::string("Image Width: " + std::to_string(image_config.image_width));
    const std::string image_height = std::string("Image Height: " + std::to_string(image_config.image_height));
    const std::string number_of_ray_bounces = std::string("Number Of Ray Bounces: " + std::to_string(image_config.number_of_ray_bounces));
    ImGui::Text(image_name.c_str());
    ImGui::Text(image_width.c_str());
    ImGui::Text(image_height.c_str());
    ImGui::Text(number_of_ray_bounces.c_str());
    const std::string temp1 = "Number of iterations left: " + std::to_string(editor_info.num_of_iterations_left);
    ImGui::Text(temp1.c_str());
    std::ostringstream temp_oss1;
    temp_oss1 << std::fixed << std::setprecision(2) << editor_info.avg_time_per_image;
    const std::string temp2 = "Average time per sample: " + temp_oss1.str() + " ms.";
    ImGui::Text(temp2.c_str());
    printEstimatedTimeOfArrival(editor_info);
    if (ImGui::Button("Stop"))
    {
        editor_commands.push_back(std::make_shared<StopGeneratingImageCommand>());
    }
}

void RayTracerGUIElement::renderImageQueueElement(RayTracerConfig& image_config)
{
    const std::string image_name = std::string("Image Name##" + std::to_string(image_config.id));
    const std::string image_width = std::string("Image Width##" + std::to_string(image_config.id));
    const std::string image_height = std::string("Image Height##" + std::to_string(image_config.id));
    const std::string samples_per_pixel = std::string("Samples Per Pixel##" + std::to_string(image_config.id));
    const std::string number_of_ray_bounces = std::string("Number Of Ray Bounces##" + std::to_string(image_config.id));
    const std::string number_of_iterations = std::string("Number Of Iterations##" + std::to_string(image_config.id));
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputText(image_name.c_str(), &image_config.image_name);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar(image_width.c_str(), ImGuiDataType_U32, &image_config.image_width);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar(image_height.c_str(), ImGuiDataType_U32, &image_config.image_height);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar(number_of_ray_bounces.c_str(), ImGuiDataType_U32, &image_config.number_of_ray_bounces);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * ray_tracing_config_input_size);
    ImGui::InputScalar(number_of_iterations.c_str(), ImGuiDataType_U32, &image_config.number_of_iterations);
    ImGui::Separator();
}

void RayTracerGUIElement::printEstimatedTimeOfArrival(const EditorInfo& editor_info)
{
    auto [eta, postfix] = adjustTimeValue(editor_info.eta);
    std::ostringstream temp_oss2;
    temp_oss2 << std::fixed << std::setprecision(2) << eta;
    const std::string temp = "ETA: " + temp_oss2.str() + postfix;
    ImGui::Text(temp.c_str());
}

std::pair<double, std::string> RayTracerGUIElement::adjustTimeValue(double time_value)
{
    std::string postfix;
    double final_value = time_value;
    if (final_value >= 3600000)
    {
        final_value /= 3600000.0;
        postfix = " h.";
    }
    else if (final_value >= 60000)
    {
        final_value /= 60000.0;
        postfix = " min.";
    }
    else if (final_value >= 1000)
    {
        final_value /= 1000.0;
        postfix = " s.";
    }
    else
    {
        postfix = " ms.";
    }

    return { final_value, postfix };
}