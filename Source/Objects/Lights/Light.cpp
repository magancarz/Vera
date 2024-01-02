#include "Light.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include "GUI/GUI.h"

Light::Light(Scene* parent_scene,
        const glm::vec3& position,
        const glm::vec3& light_direction,
        const glm::vec3& light_color,
        const glm::vec3& attenuation,
        float cutoff_angle_cosine, float cutoff_angle_outer_cosine)
    : Object(parent_scene, position), light_direction(light_direction), light_color(light_color), attenuation(attenuation),
    cutoff_angle_cosine(cutoff_angle_cosine), cutoff_angle_offset_cosine(cutoff_angle_outer_cosine)
{
    createNameForLight();
}

void Light::createNameForLight()
{
    object_id = next_id++;
    name = "light" + std::to_string(object_id);
}

void Light::prepare()
{
    createShadowMapShader();
    createShadowMapTexture();
    createLightSpaceTransform();
}

void Light::createShadowMapShader()
{
    shadow_map_shader = std::make_unique<ShadowMapShader>();
    shadow_map_shader->getAllUniformLocations();
}

void Light::prepareForShadowMapRendering()
{
    glViewport(0, 0, shadow_map_width, shadow_map_height);
    bindShadowMapTextureToFramebuffer();
    shadow_map_shader->start();
    shadow_map_shader->loadLightSpaceMatrix(light_space_transform);
}

void Light::bindShadowMapTexture() const
{
    glBindTexture(shadow_map_texture.type, shadow_map_texture.texture_id);
}

void Light::bindShadowMapTextureToFramebuffer() const
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map_texture.texture_id, 0);
}

glm::mat4 Light::getLightSpaceTransform() const
{
    return light_space_transform;
}

void Light::loadTransformationMatrixToShadowMapShader(const glm::mat4& mat)
{
    shadow_map_shader->loadTransformationMatrix(mat);
}

std::string Light::getObjectInfo()
{
    return Algorithms::vec3ToString(position) + " " + Algorithms::vec3ToString(light_color);
}

void Light::renderObjectInformationGUI()
{
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Light Details");
    ImGui::Text("Name");
    ImGui::InputText("##name", &name);

    ImGui::Separator();

    auto position_value = GUI::drawInputFieldForVector3(position, "Position");
    if (position_value.has_value())
    {
        setPosition(position_value.value());
    }

    auto light_color_value = GUI::drawColorPicker(light_color, "Light Color");
    if (light_color_value.has_value())
    {
        setLightColor(light_color_value.value());
    }
}

void Light::setPosition(const glm::vec3& value)
{
    Object::setPosition(value);
    createLightSpaceTransform();
}

glm::vec3 Light::getLightDirection() const
{
    return light_direction;
}

void Light::setLightDirection(const glm::vec3& in_light_direction)
{
    light_direction = in_light_direction;
}

glm::vec3 Light::getLightColor() const
{
    return light_color;
}

void Light::setLightColor(const glm::vec3& in_light_color)
{
    light_color = in_light_color;
}

float Light::getCutoffAngle() const
{
    return cutoff_angle_cosine;
}

void Light::setCutoffAngle(float in_cutoff_angle_cosine)
{
    cutoff_angle_cosine = in_cutoff_angle_cosine;
}

float Light::getCutoffAngleOffset() const
{
    return cutoff_angle_offset_cosine;
}

void Light::setCutoffAngleOffset(float in_cutoff_angle_offset_cosine)
{
    cutoff_angle_offset_cosine = in_cutoff_angle_offset_cosine;
}

glm::vec3 Light::getAttenuation() const
{
    return attenuation;
}

void Light::setAttenuation(const glm::vec3& in_attenuation)
{
    attenuation = in_attenuation;
}

bool Light::shouldBeOutlined() const
{
    return false;
}