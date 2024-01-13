#pragma once

#include "Shaders/ShaderProgram.h"

class LightObjectsShader : public ShaderProgram
{
public:
    LightObjectsShader();

    void getAllUniformLocations() override;

    void loadLightColor(const glm::vec3& color);

private:
    int location_light_color;
};