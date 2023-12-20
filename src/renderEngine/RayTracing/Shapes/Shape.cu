#include "Shape.h"

__device__ Shape::Shape(Object* parent, size_t id, Material* material)
    : parent(parent), id(id), material(material) {}

float Shape::calculatePDFValueOfEmittedLight(const glm::vec3& origin, const glm::vec3& direction)
{
    return 0;
}

__device__ void Shape::setTransform(glm::mat4* object_to_world_val, glm::mat4* world_to_object_val)
{
    this->object_to_world = object_to_world_val;
    this->world_to_object = world_to_object_val;
    calculateObjectBounds();
    calculateWorldBounds();
    applyTransform(*this->object_to_world);
    calculateShapeSurfaceArea();
}

__device__ void Shape::resetTransform()
{
    if (world_to_object != nullptr)
    {
        applyTransform(*world_to_object);
    }
}

__device__ glm::vec3 Shape::randomDirectionAtShape(curandState* curand_state, const glm::vec3& origin)
{
    return {0.f, 0.f, 0.f};
}

__device__ bool Shape::isEmittingLight() const
{
    return false;
}

float Shape::scatteringPDF(const HitRecord* rec, const Ray* scattered) const
{
    return 0.f;
}

glm::vec3 Shape::emitted(const glm::vec2& uv)
{
    return {0, 0, 0};
}
