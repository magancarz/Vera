#pragma once

#include <glm/ext/matrix_transform.hpp>

#include "ObjectComponent.h"

class TransformComponent : public ObjectComponent
{
public:
    explicit TransformComponent(Object& owner);

    [[nodiscard]] glm::mat4 transform() const;
    [[nodiscard]] glm::mat3 normalMatrix() const;

    glm::vec3 translation{0.f};
    glm::vec3 scale{1.f};
    glm::vec3 rotation{0.f};
};
