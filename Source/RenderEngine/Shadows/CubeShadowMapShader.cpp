#include <iostream>
#include "CubeShadowMapShader.h"

CubeShadowMapShader::CubeShadowMapShader()
    : ShaderProgram(
        "Source/RenderEngine/Shadows/cube_map_depth_vert.glsl",
        "Source/RenderEngine/Shadows/cube_map_depth_geom.glsl",
        "Source/RenderEngine/Shadows/cube_map_depth_frag.glsl") {}

void CubeShadowMapShader::loadTransformationMatrix(const glm::mat4& matrix)
{
    loadMatrix(location_transformation_matrix, matrix);
}

void CubeShadowMapShader::loadLightSpaceMatrices(const std::vector<glm::mat4>& matrices)
{
    if (matrices.size() < 6)
    {
        std::cerr << "[ERROR] Cube Shadow Map Shader: loadLightSpaceMatrices was given less than 6 matrices!\n";
    }

    for (size_t i = 0; i < 6; ++i)
    {
        loadMatrix(location_light_space_transform[i], matrices[i]);
    }
}

void CubeShadowMapShader::loadLightPosition(const glm::vec3& position)
{
    loadVector3(location_light_position, position);
}

void CubeShadowMapShader::loadFarPlane(float value)
{
    loadFloat(location_far_plane, value);
}

void CubeShadowMapShader::getAllUniformLocations()
{
    location_transformation_matrix = getUniformLocation("model");

    for (size_t i = 0; i < 6; ++i)
    {
        location_light_space_transform[i] = getUniformLocation("light_view_transforms[" + std::to_string(i) + "]");
    }

    location_light_position = getUniformLocation("light_position");
    location_far_plane = getUniformLocation("far_plane");
}