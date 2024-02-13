#pragma once

#include <glm/glm.hpp>

struct GlobalUBO
{
    glm::mat4 projection_view{};
    glm::vec3 light_direction{glm::normalize(glm::vec3{1.f, -3.f, -1.f})};
};