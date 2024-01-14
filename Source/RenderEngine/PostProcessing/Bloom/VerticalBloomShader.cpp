#include "VerticalBloomShader.h"
#include "RenderEngine/RendererDefines.h"

VerticalBloomShader::VerticalBloomShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/VerticalBlurFrag.glsl") {}

void VerticalBloomShader::getAllUniformLocations()
{
    location_blurred_texture = getUniformLocation("image");
}

void VerticalBloomShader::connectTextureUnits()
{
    loadInt(location_blurred_texture, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
}