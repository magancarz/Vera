#pragma once

#include <glm/glm.hpp>

struct CameraUBO
{
    glm::mat4 projection{};
    glm::mat4 view{};
};