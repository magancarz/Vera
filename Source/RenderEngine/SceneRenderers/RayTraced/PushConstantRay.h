#pragma once

#include <glm/glm.hpp>

struct PushConstantRay
{
    uint32_t time{0};
    uint32_t frames{0};
    uint32_t number_of_lights{0};
    float weather{0.05f};
    glm::vec3 sun_position{glm::normalize(glm::vec3{1})};
};