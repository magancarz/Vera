#include "RayTracedImageShader.h"

RayTracedImageShader::RayTracedImageShader()
{
    loadShaders("res/shaders/texture_vert.glsl", "res/shaders/texture_frag.glsl");
}
