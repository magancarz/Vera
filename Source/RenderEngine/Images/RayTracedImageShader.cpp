#include "RayTracedImageShader.h"

RayTracedImageShader::RayTracedImageShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/texture_frag.glsl") {}

void RayTracedImageShader::getAllUniformLocations()
{
    location_texture = getUniformLocation("texture_sampler");
}

void RayTracedImageShader::connectTextureUnits() const
{
    loadInt(location_texture, 0);
}