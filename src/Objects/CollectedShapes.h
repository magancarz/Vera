#pragma once

#include <vector>

struct ShapeInfo;
class Shape;

struct CollectedShapes
{
    size_t number_of_shapes{0};
    size_t number_of_light_emitting_shapes{0};
    std::vector<Shape*> shapes;
    std::vector<ShapeInfo*> shapes_infos;
    std::vector<Shape*> light_emitting_shapes;
};