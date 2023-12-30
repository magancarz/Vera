#include "ShadowMapShader.h"

ShadowMapShader::ShadowMapShader()
    : ShaderProgram("Source/RenderEngine/Shadows/simple_depth_vert.glsl", "Source/RenderEngine/Shadows/simple_frag.glsl") {}

void ShadowMapShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void ShadowMapShader::loadLightSpaceMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_light_space_transform, matrix);
}

void ShadowMapShader::getAllUniformLocations()
{
    location_light_space_transform = getUniformLocation("light_space");
    location_transformation_matrix = getUniformLocation("model");
}