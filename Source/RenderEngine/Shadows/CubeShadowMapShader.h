#pragma once

#include <vector>

#include "Shaders/ShaderProgram.h"

class CubeShadowMapShader : public ShaderProgram
{
public:
    CubeShadowMapShader();

    void loadTransformationMatrix(const glm::mat4& matrix);
    void loadLightSpaceMatrices(const std::vector<glm::mat4>& matrices);
    void loadLightPosition(const glm::vec3& position);
    void loadFarPlane(float value);

    void getAllUniformLocations() override;

private:
    unsigned int location_transformation_matrix;
    unsigned int location_light_space_transform[6];
    unsigned int location_light_position;
    unsigned int location_far_plane;
};