#pragma once

#include <memory>
#include <vector>

#include "CollectedShapes.h"

class Object;

class ShapesCollector 
{
public:
    ShapesCollector(const std::vector<std::shared_ptr<Object>>& objects_to_collect_from);

    CollectedShapes collectShapes();

private:
    void findNumberOfAllShapes();
    void collectShapesFromObject(const std::shared_ptr<Object>& object);

    std::vector<std::shared_ptr<Object>> objects_to_collect_from;
    CollectedShapes collected_shapes{};
};