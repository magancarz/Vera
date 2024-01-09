#include "OutlineShader.h"

OutlineShader::OutlineShader()
    : ShaderProgram("Resources/Shaders/outlineVert.glsl", "Resources/Shaders/outlineFrag.glsl") {}

void OutlineShader::getAllUniformLocations()
{
    location_color = getUniformLocation("color");
}

void OutlineShader::loadOutlineColor(const glm::vec3& color) const
{
    loadVector3(location_color, color);
}