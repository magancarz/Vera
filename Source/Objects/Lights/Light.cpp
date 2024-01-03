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
    shadow_map_texture = std::make_shared<utils::Texture>();
}

void Light::createShadowMapTexture()
{
    shadow_map_texture->bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_map_width, shadow_map_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
}

void Light::prepareForShadowMapRendering()
{
    glViewport(0, 0, shadow_map_width, shadow_map_height);
    bindShadowMapTextureToFramebuffer();
    shadow_map_shader->start();
}

void Light::bindShadowMapTexture() const
{
    shadow_map_texture->bindTexture();
}

void Light::bindShadowMapTextureToFramebuffer() const
{
    shadow_map_texture->bindToCurrentFramebufferAsDepthAttachment();
}

glm::mat4 Light::getLightSpaceTransform() const
{
    return light_space_transform;
}

std::shared_ptr<utils::Texture> Light::getShadowMap() const
{
    return shadow_map_texture;
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