#pragma once

#include <glm/ext/matrix_transform.hpp>

class TransformComponent
{
public:
    glm::vec3 translation{};
    glm::vec3 scale{1.f};
    glm::vec3 rotation{};

    glm::mat4 transform();
    glm::mat3 normalMatrix();
};
