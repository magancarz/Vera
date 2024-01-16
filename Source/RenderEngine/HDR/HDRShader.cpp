#include "ToneMappingShader.h"
#include "RenderEngine/RendererDefines.h"

ToneMappingShader::ToneMappingShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/HDRFrag.glsl") {}

void ToneMappingShader::getAllUniformLocations()
{
    location_hdr_buffer = getUniformLocation("hdr_buffer");
}

void ToneMappingShader::connectTextureUnits()
{
    loadInt(location_hdr_buffer, RendererDefines::HDR_BUFFER_INDEX);
}