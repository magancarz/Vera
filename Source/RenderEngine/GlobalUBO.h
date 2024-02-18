#pragma once

#include <glm/glm.hpp>

#include "RendererDefines.h"

struct GlobalUBO
{
    glm::mat4 projection{};
    glm::mat4 view{};
    glm::mat4 inverse_view{};
    glm::vec4 ambient_light_color{1.f, 1.f, 1.f, 0.02f}; // w is intensity
    PointLight point_lights[RendererDefines::MAX_NUMBER_OF_LIGHTS];
    int number_of_lights;
};