#pragma once

#include "BVHTreeBuilder.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"

class BVHTreeTraverser
{
public:
    BVHTreeTraverser(Triangle** shapes, BVHTreeNode* nodes)
        : shapes(shapes), nodes(nodes) {}

    __device__ HitRecord checkIntersection(const Ray* ray) const;

protected:
    Triangle** shapes;
    BVHTreeNode* nodes{nullptr};
};