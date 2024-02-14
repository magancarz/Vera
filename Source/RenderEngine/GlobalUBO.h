#pragma once

#include <glm/glm.hpp>

struct GlobalUBO
{
    glm::mat4 projection{};
    glm::mat4 view{};
    glm::vec4 ambient_light_color{1.f, 1.f, 1.f, 0.02f}; // w is intensity
    glm::vec3 light_position{-1.f};
    alignas(16) glm::vec4 light_color{1.f}; // w is intensity
};