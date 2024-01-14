#include "HorizontalBloomShader.h"
#include "RenderEngine/RendererDefines.h"

HorizontalBloomShader::HorizontalBloomShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/HorizontalBlurFrag.glsl") {}

void HorizontalBloomShader::getAllUniformLocations()
{
    location_blurred_texture = getUniformLocation("image");
}

void HorizontalBloomShader::connectTextureUnits()
{
    loadInt(location_blurred_texture, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
}