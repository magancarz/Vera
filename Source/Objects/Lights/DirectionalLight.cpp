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

void DirectionalLight::createShadowMapShader()
{
    shadow_map_shader = std::make_unique<ShadowMapShader>();
    shader_program_as_shadow_map_shader = dynamic_cast<ShadowMapShader*>(shadow_map_shader.get());
}

void DirectionalLight::loadTransformationMatrixToShadowMapShader(const glm::mat4& mat)
{
    shader_program_as_shadow_map_shader->loadTransformationMatrix(mat);
}

void DirectionalLight::createLightSpaceTransform()
{
    glm::mat4 light_projection = glm::ortho(
            -shadow_map_orthographic_projection_x_span, shadow_map_orthographic_projection_x_span,
            -shadow_map_orthographic_projection_y_span, shadow_map_orthographic_projection_y_span,
            near_plane, far_plane);

    glm::mat4 light_view = glm::lookAt(-light_direction * 10.f, {0, 0, 0}, {0, 0, 1});

    light_space_transform = light_projection * light_view;
    shader_program_as_shadow_map_shader->loadLightSpaceMatrix(light_space_transform);
}

int DirectionalLight::getLightType() const
{
    return 0;
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