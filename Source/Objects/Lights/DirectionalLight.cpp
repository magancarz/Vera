#include "DirectionalLight.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include "GUI/GUI.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

DirectionalLight::DirectionalLight(Scene* parent_scene)
    : Light(parent_scene) {}

DirectionalLight::DirectionalLight(Scene* parent_scene, const glm::vec3& light_direction, const glm::vec3& light_color)
    : Light(parent_scene, {0, 0, 0}, light_direction, light_color) {}

void DirectionalLight::createShadowMapTexture()
{
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture.texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_width, shadow_map_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
}

void DirectionalLight::createLightSpaceTransform()
{
    glm::mat4 light_projection = glm::ortho(
            -shadow_map_orthographic_projection_x_span, shadow_map_orthographic_projection_x_span,
            -shadow_map_orthographic_projection_y_span, shadow_map_orthographic_projection_y_span,
            near_plane, far_plane);

    glm::mat4 light_view = glm::lookAt(-light_direction * 10.f, {0, 0, 0}, {0, 0, 1});

    light_space_transform = light_projection * light_view;
}

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
        Light::setPosition(light_direction_value.value());
    }
}