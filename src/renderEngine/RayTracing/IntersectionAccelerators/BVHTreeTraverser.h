#pragma once

#include "BVHTreeBuilder.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"

class BVHTreeTraverser
{
public:
    BVHTreeTraverser(Shape** shapes, BVHTreeNode* nodes)
        : shapes(shapes), nodes(nodes) {}

    __device__ HitRecord checkIntersection(const Ray* ray) const;

protected:
    Shape** shapes;
    BVHTreeNode* nodes{nullptr};
};