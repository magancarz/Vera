#pragma once

#include <glm/glm.hpp>

struct GlobalUBO
{
    glm::mat4 projection{};
    glm::mat4 view{};
    glm::mat4 inverse_view{};
};