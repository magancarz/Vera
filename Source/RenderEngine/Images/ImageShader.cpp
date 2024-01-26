#include "ImageShader.h"

ImageShader::ImageShader()
    : ShaderProgram("Resources/Shaders/QuadVert.glsl", "Resources/Shaders/texture_frag.glsl") {}

void ImageShader::getAllUniformLocations()
{
    location_texture = getUniformLocation("texture_sampler");
}

void ImageShader::connectTextureUnits()
{
    loadInt(location_texture, 0);
}