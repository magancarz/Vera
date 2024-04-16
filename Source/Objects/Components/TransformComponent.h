#pragma once

#include <glm/ext/matrix_transform.hpp>
#include "RenderEngine/RenderingAPI/Buffer.h"

class TransformComponent
{
public:
    glm::mat4 transform();
    glm::mat3 normalMatrix();

    glm::vec3 translation{};
    glm::vec3 scale{1.f};
    glm::vec3 rotation{};

    std::unique_ptr<Buffer> transform_buffer;
};
