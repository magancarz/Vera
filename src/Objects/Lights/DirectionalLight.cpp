#include "DirectionalLight.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include "GUI/GUI.h"

DirectionalLight::DirectionalLight(Scene* parent_scene)
    : Light(parent_scene) {}

DirectionalLight::DirectionalLight(Scene* parent_scene, const glm::vec3& light_direction, const glm::vec3& light_color)
    : Light(parent_scene, {0, 0, 0}, light_direction, light_color) {}

std::string DirectionalLight::getObjectInfo()
{
    return DIRECTIONAL_LIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec3ToString(light_direction);
}

void DirectionalLight::renderObjectInformationGUI()
{
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Light Details");
    ImGui::Text("Name");
    ImGui::InputText("##name", &name);

    ImGui::Separator();

    auto light_color_value = GUI::drawColorPicker(light_color, "Light Color");
    if (light_color_value.has_value())
    {
        setLightColor(light_color_value.value());
    }

    auto light_direction_value = GUI::drawInputFieldForVector3(light_direction, "Light Direction");
    if (light_direction_value.has_value())
    {
        setLightDirection(light_direction_value.value());
    }
}