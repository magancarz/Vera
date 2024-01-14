#pragma once

#include "Shaders/ShaderProgram.h"

class SSAOShader : public ShaderProgram
{
public:
    SSAOShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

    void loadSamples(const std::vector<glm::vec3>& samples);

private:
    int location_g_position;
    int location_g_normal;
    int location_noise_texture;
    int location_samples[64];
};
