#include "SSAOBlurShader.h"
#include "RenderEngine/RendererDefines.h"

SSAOBlurShader::SSAOBlurShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/SSAOBlurFrag.glsl") {}

void SSAOBlurShader::getAllUniformLocations()
{
    location_ssao_input = getUniformLocation("ssao_input");
}

void SSAOBlurShader::connectTextureUnits()
{
    loadInt(location_ssao_input, RendererDefines::G_BUFFER_STARTING_INDEX + RendererDefines::NUMBER_OF_G_BUFFER_TEXTURES + 0);
}