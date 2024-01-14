#include "SSAOShader.h"
#include "RenderEngine/RendererDefines.h"

SSAOShader::SSAOShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/SSAOFrag.glsl") {}

void SSAOShader::getAllUniformLocations()
{
    location_g_position = getUniformLocation("g_position");
    location_g_normal = getUniformLocation("g_normal");
    location_noise_texture = getUniformLocation("noise_texture");
    for(size_t i = 0; i < 64; ++i)
    {
        location_samples[i] = getUniformLocation("samples[" + std::to_string(i) + "]");
    }
}

void SSAOShader::connectTextureUnits()
{
    loadInt(location_g_position, RendererDefines::G_BUFFER_STARTING_INDEX + 0);
    loadInt(location_g_normal, RendererDefines::G_BUFFER_STARTING_INDEX + 1);
    loadInt(location_noise_texture, RendererDefines::G_BUFFER_STARTING_INDEX + RendererDefines::NUMBER_OF_G_BUFFER_TEXTURES + 0);
}

void SSAOShader::loadSamples(const std::vector<glm::vec3>& samples)
{
    for (size_t i = 0; i < 64; ++i)
    {
        loadVector3(location_samples[i], samples[i]);
    }
}