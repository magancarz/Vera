#pragma once

#include <glm/glm.hpp>

struct LightInfo
{
public:
    LightInfo(const glm::vec3& light_position, const glm::vec3& light_color, const glm::vec3& attenuation)
        : light_position(light_position), light_color(light_color), attenuation(attenuation) {}

private:
    glm::vec3 light_position;
    uint32_t pad0{0};

    glm::vec3 light_color;
    uint32_t pad1{0};

    glm::vec3 attenuation;
    uint32_t pad2{0};
};