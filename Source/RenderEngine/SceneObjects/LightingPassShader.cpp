#include "LightingPassShader.h"

#include "RenderEngine/RendererDefines.h"

LightingPassShader::LightingPassShader()
    : ShaderProgram("Resources/Shaders/LightingPassVert.glsl", "Resources/Shaders/LightingPassFrag.glsl") {}

void LightingPassShader::getAllUniformLocations()
{
    location_g_position = getUniformLocation("g_position");
    location_g_normal = getUniformLocation("g_normal");
    location_g_color_spec = getUniformLocation("g_color_spec");
    location_view_position = getUniformLocation("view_position");
}

void LightingPassShader::connectTextureUnits()
{
    loadInt(location_g_position, RendererDefines::G_BUFFER_STARTING_INDEX + 0);
    loadInt(location_g_normal, RendererDefines::G_BUFFER_STARTING_INDEX + 1);
    loadInt(location_g_color_spec, RendererDefines::G_BUFFER_STARTING_INDEX + 2);
}

void LightingPassShader::loadViewPosition(const glm::vec3& position)
{
    loadVector3(location_view_position, position);
}