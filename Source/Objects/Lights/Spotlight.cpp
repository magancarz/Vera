#include "Spotlight.h"
#include "GUI/GUI.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"

Spotlight::Spotlight(Scene* parent_scene)
    : Light(parent_scene, {0, 0, 0}, {0, -1, 0}, {1, 1, 1}, {1, 0.01, 0.0001}, 0.9f, 0.85f) {}

Spotlight::Spotlight(Scene* parent_scene, const glm::vec3& position, const glm::vec3& light_direction, const glm::vec3& light_color,
        const glm::vec3& attenuation, float cutoff_angle, float cutoff_angle_outer)
    : Light(parent_scene, position, light_direction, light_color, attenuation, cutoff_angle, cutoff_angle_outer) {}

void Spotlight::createShadowMapShader()
{
    shadow_map_shader = std::make_unique<ShadowMapShader>();
    shadow_map_shader->getAllUniformLocations();
    shader_program_as_shadow_map_shader = dynamic_cast<ShadowMapShader*>(shadow_map_shader.get());
}

void Spotlight::loadTransformationMatrixToShadowMapShader(const glm::mat4& mat)
{
    shader_program_as_shadow_map_shader->loadTransformationMatrix(mat);
}

int Spotlight::getLightType() const
{
    return 2;
}

std::string Spotlight::getObjectInfo()
{
    return SPOTLIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec3ToString(light_direction) + " " + Algorithms::vec3ToString(attenuation) + " " +
           Algorithms::floatToString(cutoff_angle_cosine) + " " + Algorithms::floatToString(cutoff_angle_offset_cosine);
}

void Spotlight::renderObjectInformationGUI()
{
    Light::renderObjectInformationGUI();

    auto light_direction_value = GUI::drawInputFieldForVector3(light_direction, "Light Direction");
    if (light_direction_value.has_value())
    {
        setLightDirection(light_direction_value.value());
    }

    auto attenuation_value = GUI::drawInputFieldForVector3(attenuation, "Attenuation");
    if (attenuation_value.has_value())
    {
        setAttenuation(attenuation_value.value());
    }

    GUI::drawInputFieldForFloat(&cutoff_angle_cosine, "Cutoff Angle Cosine");
    GUI::drawInputFieldForFloat(&cutoff_angle_offset_cosine, "Cutoff Angle Offset Cosine");
}

void Spotlight::createLightSpaceTransform()
{
    glm::mat4 light_projection = glm::perspective(
        glm::acos(cutoff_angle_offset_cosine) * 2,
        static_cast<float>(shadow_map_width) / static_cast<float>(shadow_map_height),
        near_plane, far_plane);

    glm::mat4 light_view = glm::lookAt(position, position + light_direction, {0, 0, 1});

    light_space_transform = light_projection * light_view;
    shader_program_as_shadow_map_shader->start();
    shader_program_as_shadow_map_shader->loadLightSpaceMatrix(light_space_transform);
    ShaderProgram::stop();
}