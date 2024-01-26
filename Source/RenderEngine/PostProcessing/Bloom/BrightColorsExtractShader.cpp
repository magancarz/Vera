#include "BrightColorsExtractShader.h"

#include "RenderEngine/RendererDefines.h"

BrightColorsExtractShader::BrightColorsExtractShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/BrightColorsExtractShaderFrag.glsl") {}

void BrightColorsExtractShader::getAllUniformLocations()
{
    location_hdr_color_buffer = getUniformLocation("hdr_color_buffer");
}

void BrightColorsExtractShader::connectTextureUnits()
{
    loadInt(location_hdr_color_buffer, RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
}