#pragma once

#include "Shaders/ShaderProgram.h"

class ShadowMapShader : public ShaderProgram
{
public:
    ShadowMapShader();

    void loadTransformationMatrix(const glm::mat4& matrix) const;
    void loadLightSpaceMatrix(const glm::mat4& matrix) const;

    void getAllUniformLocations() override;

private:
    unsigned int location_light_space_transform;
    unsigned int location_transformation_matrix;
};