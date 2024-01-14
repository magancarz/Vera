#include "HDRShader.h"
#include "RenderEngine/RendererDefines.h"

HDRShader::HDRShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/HDRFrag.glsl") {}

void HDRShader::getAllUniformLocations()
{
    location_hdr_buffer = getUniformLocation("hdr_buffer");
}

void HDRShader::connectTextureUnits()
{
    loadInt(location_hdr_buffer, RendererDefines::HDR_BUFFER_INDEX);
}