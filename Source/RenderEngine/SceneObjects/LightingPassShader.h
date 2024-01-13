#pragma once

#include "Shaders/ShaderProgram.h"

class LightingPassShader : public ShaderProgram
{
public:
    LightingPassShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

    void loadViewPosition(const glm::vec3& position);

private:
    int location_g_position;
    int location_g_normal;
    int location_g_color_spec;
    int location_shadow_map;
    int location_view_position;
};