#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class OutlineShader : public ShaderProgram
{
public:
    OutlineShader();
    virtual ~OutlineShader() = default;

    void getAllUniformLocations() override;

    void loadOutlineColor(const glm::vec3& color);

private:
    int location_color;
};
