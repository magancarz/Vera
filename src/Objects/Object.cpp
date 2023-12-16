#include "Object.h"

#include <string>
#include <glm/gtx/transform.hpp>

#include "Materials/MaterialAsset.h"
#include "Scene/Scene.h"
#include "ObjectInfo.h"
#include "Materials/Material.h"
#include "Utils/CurandUtils.h"
#include "RenderEngine/RayTracing/Shapes/ShapeInfo.h"

Object::Object(
    Scene* parent_scene,
    std::shared_ptr<MaterialAsset> material,
    std::shared_ptr<RawModel> model_data,
    const glm::vec3& position,
    const glm::vec3& rotation,
    float scale)
    : parent_scene(parent_scene),
      material(std::move(material)),
      model_data(std::move(model_data)),
      position(position),
      rotation(rotation),
      scale(scale),
      num_of_triangles(this->model_data->triangles.size())
{
    cuda_shapes = dmm::DeviceMemoryPointer<Triangle>(num_of_triangles);
    cuda_triangles_infos = dmm::DeviceMemoryPointer<ShapeInfo>(num_of_triangles);
    is_emitting_some_light = this->material->material->isEmittingLight();
    createObjectName();
    createMesh();
}

void Object::createObjectName()
{
    object_id = next_id;
    name = "object" + std::to_string(object_id);
    ++next_id;
}

void Object::changeMaterial(Material* material)
{
    triangles_emitting_light.clear();
    for (int i = 0; i < num_of_triangles; ++i)
    {
        cuda_shapes[i].material = material;
        if (cuda_shapes[i].isEmittingLight())
        {
            triangles_emitting_light.push_back(&cuda_shapes[i]);
        }
    }

    is_emitting_some_light = !triangles_emitting_light.empty();
}

int Object::getNumberOfTriangles()
{
    return num_of_triangles;
}

std::vector<Triangle*> Object::getTrianglesEmittingLight() const
{
    return triangles_emitting_light;
}

void Object::createMesh()
{
    allocateShapesOnDeviceMemory();
    refreshTriangles();
    changeMaterial(material);
}

void Object::allocateShapesOnDeviceMemory()
{
    for (int i = 0; i < num_of_triangles; ++i)
    {
        cuda_shapes[i].prepare(model_data->triangles[i]);
        cuda_shapes[i].parent = this;
        cuda_shapes[i].id = next_triangle_id + i;
    }
    next_triangle_id += num_of_triangles;
}

void Object::refreshTriangles()
{
    for (int i = 0; i < num_of_triangles; ++i)
    {
        cuda_shapes[i].resetTransform();
        cuda_triangles_infos[i].world_bounds = cuda_shapes[i].world_bounds;
    }

    const glm::mat4 object_to_world = createObjectToWorldTransform();
    this->object_to_world.copyFrom(&object_to_world);
    const glm::mat4 world_to_object = glm::inverse(object_to_world);
    this->world_to_object.copyFrom(&world_to_object);
    
    for (int i = 0; i < num_of_triangles; ++i)
    {
        cuda_shapes[i].setTransform(this->object_to_world.data(), this->world_to_object.data());
        cuda_triangles_infos[i].world_bounds = cuda_shapes[i].world_bounds;
    }
}

glm::mat4 Object::createObjectToWorldTransform()
{
    glm::mat4 transformation_matrix = translate(glm::mat4(1.0f), position);
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.x), glm::vec3(1, 0, 0));
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.y), glm::vec3(0, 1, 0));
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.z), glm::vec3(0, 0, 1));
    transformation_matrix = glm::scale(transformation_matrix, glm::vec3(scale));

    return transformation_matrix;
}

glm::mat4 Object::createWorldToObjectTransform()
{
    glm::mat4 transformation_matrix = createObjectToWorldTransform();
    transformation_matrix = glm::inverse(transformation_matrix);

    return transformation_matrix;
}

void Object::refreshObject()
{
    refreshTriangles();
    parent_scene->notifyOnObjectChange();
}

void Object::changeMaterial(std::shared_ptr<MaterialAsset> new_material)
{
    changeMaterial(new_material->cuda_material.data());
    material = std::move(new_material);
    parent_scene->notifyOnObjectMaterialChange(this);
}

void Object::increasePosition(const float dx, const float dy, const float dz)
{
    position.x += dx;
    position.y += dy;
    position.z += dz;
}

void Object::increaseRotation(const float rx, const float ry, const float rz)
{
    rotation.x += rx;
    rotation.y += ry;
    rotation.z += rz;
}

glm::vec3 Object::getPosition() const
{
    return position;
}

glm::vec3 Object::getRotation() const
{
    return rotation;
}

float Object::getScale() const
{
    return scale;
}

void Object::setPosition(const glm::vec3& value)
{
    position = value;
    refreshObject();
}

void Object::setRotation(const glm::vec3& value)
{
    rotation = value;
    refreshObject();
}

void Object::setScale(float value)
{
    scale = value;
    refreshObject();
}

void Object::setShouldBeOutlined(bool value)
{
    should_be_outlined = value;
}

bool Object::shouldBeOutlined() const
{
    return should_be_outlined;
}

Material* Object::getMaterial() const
{
    return material->cuda_material.data();
}

std::shared_ptr<RawModel> Object::getModelData() const
{
    return model_data;
}

Triangle* Object::getTriangles()
{
    return cuda_shapes.data();
}

ObjectInfo Object::getObjectInfo()
{
    return { name, model_data->model_name, material->material->name, position, rotation, scale };
}

bool Object::operator==(unsigned id) const
{
    return object_id == id;
}

bool Object::isTriangleEmittingLight(int triangle_index) const
{
	return cuda_shapes[triangle_index].isEmittingLight();
}
