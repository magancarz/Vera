#pragma once

#include "glm/vec3.hpp"

struct Vertex;

struct AABB
{
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::min()};

    static AABB merge(const AABB& first, const AABB& second);
    static AABB merge(const AABB& first, const glm::vec3& point);
    static AABB fromTriangle(const Vertex& first, const Vertex& second, const Vertex& third);

    [[nodiscard]] float surfaceArea() const;
    [[nodiscard]] size_t maximumExtent() const;
    [[nodiscard]] bool contains(const glm::vec3& point) const;
};
