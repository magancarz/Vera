#include "ShapesCollector.h"

#include "Object.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"

ShapesCollector::ShapesCollector(const std::vector<std::shared_ptr<Object>>& objects_to_collect_from)
    : objects_to_collect_from{objects_to_collect_from} {}

CollectedShapes ShapesCollector::collectShapes()
{
    findNumberOfAllShapes();

    for (const auto& object : objects_to_collect_from)
    {
        collectShapesFromObject(object);
    }

    return collected_shapes;
}

void ShapesCollector::findNumberOfAllShapes()
{
    for (const auto& object : objects_to_collect_from)
    {
        collected_shapes.number_of_shapes += object->getNumberOfShapes();
        collected_shapes.number_of_light_emitting_shapes += object->getNumberOfLightEmittingShapes();
    }

    collected_shapes.shapes.reserve(collected_shapes.number_of_shapes);
    collected_shapes.light_emitting_shapes.reserve(collected_shapes.number_of_light_emitting_shapes);
}

void ShapesCollector::collectShapesFromObject(const std::shared_ptr<Object>& object)
{
    Shape** object_shapes = object->getShapes();
    ShapeInfo* shapes_infos = object->getShapesInfos();
    for (size_t i = 0; i < object->getNumberOfShapes(); ++i)
    {
        collected_shapes.shapes.push_back(object_shapes[i]);
        collected_shapes.shapes_infos.push_back(&shapes_infos[i]);
    }

    for (const auto& shape : object->getShapesEmittingLight())
    {
        collected_shapes.light_emitting_shapes.push_back(shape);
    }
}
