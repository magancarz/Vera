#include "RayTracedImageShader.h"

RayTracedImageShader::RayTracedImageShader()
{
    loadShaders("Resources/Shaders/texture_vert.glsl", "Resources/Shaders/texture_frag.glsl");
}
