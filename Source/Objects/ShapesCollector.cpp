#include "ShapesCollector.h"

#include "Objects/TriangleMesh.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"

ShapesCollector::ShapesCollector(const std::vector<std::weak_ptr<TriangleMesh>>& objects_to_collect_from)
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
        collected_shapes.number_of_shapes += object.lock()->getNumberOfShapes();
        collected_shapes.number_of_light_emitting_shapes += object.lock()->getNumberOfLightEmittingShapes();
    }

    collected_shapes.shapes.reserve(collected_shapes.number_of_shapes);
    collected_shapes.light_emitting_shapes.reserve(collected_shapes.number_of_light_emitting_shapes);
}

void ShapesCollector::collectShapesFromObject(const std::weak_ptr<TriangleMesh>& object)
{
    Shape** object_shapes = object.lock()->getShapes();
    ShapeInfo* shapes_infos = object.lock()->getShapesInfos();
    for (size_t i = 0; i < object.lock()->getNumberOfShapes(); ++i)
    {
        collected_shapes.shapes.push_back(object_shapes[i]);
        collected_shapes.shapes_infos.push_back(&shapes_infos[i]);
    }

    for (const auto& shape : object.lock()->getShapesEmittingLight())
    {
        collected_shapes.light_emitting_shapes.push_back(shape);
    }
}
