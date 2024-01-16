#include "VerticalBlurShader.h"
#include "RenderEngine/RendererDefines.h"

VerticalBlurShader::VerticalBlurShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/VerticalBlurFrag.glsl") {}

void VerticalBlurShader::getAllUniformLocations()
{
    location_blurred_texture = getUniformLocation("image");
}

void VerticalBlurShader::connectTextureUnits()
{
    loadInt(location_blurred_texture, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
}