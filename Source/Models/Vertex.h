#pragma once

#include "glm/glm.hpp"

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texture_coordinate;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    bool operator==(const Vertex& other) const
    {
        return position == other.position && normal == other.normal && texture_coordinate == other.texture_coordinate;
    }
};