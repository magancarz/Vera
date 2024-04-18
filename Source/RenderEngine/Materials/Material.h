#pragma once

#include <glm/glm.hpp>

struct Material
{
    alignas(16) glm::vec3 color{};
    int brightness{0};

    //TODO: remember to change this
//    uint32_t material_index{0};
};