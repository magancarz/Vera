#include "LightObjectsShader.h"

LightObjectsShader::LightObjectsShader()
    : ShaderProgram("Resources/Shaders/LightObjectsVert.glsl", "Resources/Shaders/LightObjectsFrag.glsl") {}

void LightObjectsShader::getAllUniformLocations()
{
    location_light_color = getUniformLocation("light_color");
}

void LightObjectsShader::loadLightColor(const glm::vec3& color)
{
    loadVector3(location_light_color, color);
}