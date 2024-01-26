#include "HorizontalBlurShader.h"
#include "RenderEngine/RendererDefines.h"

HorizontalBlurShader::HorizontalBlurShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/HorizontalBlurFrag.glsl") {}

void HorizontalBlurShader::getAllUniformLocations()
{
    location_blurred_texture = getUniformLocation("image");
}

void HorizontalBlurShader::connectTextureUnits()
{
    loadInt(location_blurred_texture, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
}