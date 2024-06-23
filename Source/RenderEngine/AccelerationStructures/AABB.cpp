#include "AABB.h"

#include "Assets/Model/Vertex.h"

AABB AABB::merge(const AABB& first, const AABB& second)
{
    AABB merged;
    merged.min = glm::min(first.min, second.min);
    merged.max = glm::max(first.max, second.max);

    return merged;
}

AABB AABB::merge(const AABB& first, const glm::vec3& point)
{
    AABB merged;
    merged.min = glm::min(first.min, point);
    merged.max = glm::max(first.max, point);

    return merged;
}

AABB AABB::fromTriangle(const Vertex& first, const Vertex& second, const Vertex& third)
{
    AABB triangle_aabb;
    triangle_aabb.min = glm::min(first.position, glm::min(second.position, third.position));
    triangle_aabb.max = glm::max(first.position, glm::max(second.position, third.position));

    return triangle_aabb;
}

float AABB::surfaceArea() const
{
    float a = max.x - min.x;
    float b = max.y - min.y;
    float c = max.z - min.z;

    return 2.0f * (a * b + a * c + b * c);
}

size_t AABB::maximumExtent() const
{
    float x_extent = glm::abs(max.x - min.x);
    float y_extent = glm::abs(max.y - min.y);
    float z_extent = glm::abs(max.z - min.z);
    return x_extent > y_extent ? (x_extent > z_extent ? 0 : 2) : (y_extent > z_extent ? 1 : 2);
}

bool AABB::contains(const glm::vec3& point) const
{
    return point.x <= max.x && point.x >= min.x &&
           point.y <= max.y && point.y >= min.y &&
           point.y <= max.y && point.y >= min.y;
}
