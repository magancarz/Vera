#include "Shape.h"

__device__ Shape::Shape(Object* parent, size_t id, Material* material)
    : parent(parent), id(id), material(material) {}

__device__ void Shape::setTransform(glm::mat4* object_to_world_val, glm::mat4* world_to_object_val)
{
    resetTransform();
    this->object_to_world = object_to_world_val;
    this->world_to_object = world_to_object_val;
    calculateObjectBounds();
    calculateWorldBounds();
    calculateShapeSurfaceArea();
    applyTransform(*this->object_to_world);
}

__device__ void Shape::resetTransform()
{
    if (world_to_object != nullptr)
    {
        applyTransform(*world_to_object);
    }
}
