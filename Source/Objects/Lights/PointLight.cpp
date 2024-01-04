#include "PointLight.h"
#include "GUI/GUI.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

PointLight::PointLight(Scene* parent_scene)
    : Light(parent_scene) {}

PointLight::PointLight(Scene* parent_scene, const glm::vec3& position, const glm::vec3& light_color, const glm::vec3& attenuation)
    : Light(parent_scene, position, {0, -1, 0}, light_color, attenuation) {}

void PointLight::createShadowMapShader()
{
    shadow_map_shader = std::make_unique<CubeShadowMapShader>();
    shader_program_as_cube_shadow_map_shader = dynamic_cast<CubeShadowMapShader*>(shadow_map_shader.get());
    shader_program_as_cube_shadow_map_shader->getAllUniformLocations();
    shader_program_as_cube_shadow_map_shader->loadFarPlane(far_plane);
    shader_program_as_cube_shadow_map_shader->loadLightPosition(position);
}

void PointLight::loadTransformationMatrixToShadowMapShader(const glm::mat4& mat)
{
    shader_program_as_cube_shadow_map_shader->loadTransformationMatrix(mat);
}

void PointLight::setPosition(const glm::vec3& value)
{
    Light::setPosition(value);

    shader_program_as_cube_shadow_map_shader->loadLightPosition(position);
}

void PointLight::createShadowMapTexture()
{
    shadow_map_texture = std::make_shared<utils::Texture>();
    shadow_map_texture->setAsCubeMapTexture();
    shadow_map_texture->bindTexture();
    for (size_t i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT,
             shadow_map_width, shadow_map_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}

void PointLight::createLightSpaceTransform()
{
    float aspect = static_cast<float>(shadow_map_width) / static_cast<float>(shadow_map_height);
    glm::mat4 light_projection = glm::perspective(
        glm::radians(90.0f),
        aspect,
        near_plane, far_plane
    );

    light_view_transforms[0] = light_projection * glm::lookAt(position, position + glm::vec3(1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0));
    light_view_transforms[1] = light_projection * glm::lookAt(position, position + glm::vec3(-1.0,0.0,0.0), glm::vec3(0.0,-1.0,0.0));
    light_view_transforms[2] = light_projection * glm::lookAt(position, position + glm::vec3(0.0,1.0,0.0), glm::vec3(0.0,0.0,1.0));
    light_view_transforms[3] = light_projection * glm::lookAt(position, position + glm::vec3(0.0,-1.0,0.0), glm::vec3(0.0,0.0,-1.0));
    light_view_transforms[4] = light_projection * glm::lookAt(position, position + glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,-1.0,0.0));
    light_view_transforms[5] = light_projection * glm::lookAt(position, position + glm::vec3(0.0,0.0,-1.0), glm::vec3(0.0,-1.0,0.0));

    shader_program_as_cube_shadow_map_shader->loadLightSpaceMatrices(light_view_transforms);
}

int PointLight::getLightType() const
{
    return 1;
}

std::string PointLight::getObjectInfo()
{
    return POINT_LIGHT_TAG + " " + Light::getObjectInfo() + " " + Algorithms::vec3ToString(attenuation);
}

void PointLight::renderObjectInformationGUI()
{
    Light::renderObjectInformationGUI();

    auto attenuation_value = GUI::drawInputFieldForVector3(attenuation, "Attenuation");
    if (attenuation_value.has_value())
    {
        setAttenuation(attenuation_value.value());
    }
}