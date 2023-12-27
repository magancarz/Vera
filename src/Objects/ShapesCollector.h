#pragma once

#include <memory>
#include <vector>

#include "CollectedShapes.h"

class TriangleMesh;

class ShapesCollector 
{
public:
    ShapesCollector(const std::vector<std::weak_ptr<TriangleMesh>>& objects_to_collect_from);

    CollectedShapes collectShapes();

private:
    void findNumberOfAllShapes();
    void collectShapesFromObject(const std::weak_ptr<TriangleMesh>& object);

    std::vector<std::weak_ptr<TriangleMesh>> objects_to_collect_from;
    CollectedShapes collected_shapes{};
};