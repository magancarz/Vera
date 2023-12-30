#pragma once

#include "BVHTreeBuilder.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"

class BVHTreeTraverser
{
public:
    BVHTreeTraverser(dmm::DeviceMemoryPointer<Shape*> shapes, dmm::DeviceMemoryPointer<BVHTreeNode> nodes)
        : shapes(std::move(shapes)), nodes(std::move(nodes)) {}

    __device__ HitRecord checkIntersection(const Ray* ray) const;

protected:
    dmm::DeviceMemoryPointer<Shape*> shapes;
    dmm::DeviceMemoryPointer<BVHTreeNode> nodes;
};