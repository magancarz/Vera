#include "CombineShader.h"
#include "RenderEngine/RendererDefines.h"

CombineShader::CombineShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/CombineFrag.glsl") {}

void CombineShader::getAllUniformLocations()
{
    location_blurred_texture = getUniformLocation("blurred_texture");
    location_hdr_color_buffer = getUniformLocation("hdr_color_buffer");
}

void CombineShader::connectTextureUnits()
{
    loadInt(location_blurred_texture, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    loadInt(location_hdr_color_buffer, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 1);
}