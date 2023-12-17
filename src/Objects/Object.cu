#include "Object.h"

#include <string>
#include <glm/gtx/transform.hpp>

#include "Materials/MaterialAsset.h"
#include "Scene/Scene.h"
#include "ObjectInfo.h"
#include "Materials/Material.h"
#include "Utils/CurandUtils.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"

namespace cudaObjectUtils
{
    __device__ void transformTriangle(
        glm::mat4* object_to_world, glm::mat4* world_to_object,
        Shape* shape, ShapeInfo* shapes_info)
    {
        shape->setTransform(object_to_world, world_to_object);
        shapes_info->world_bounds = shape->world_bounds;
    }

    __global__ void createTrianglesOnDeviceMemory(
        Object* parent,
        glm::mat4* object_to_world, glm::mat4* world_to_object,
        const TriangleData* shapes_data, Shape** cuda_shapes,
        ShapeInfo* shapes_infos,
        Material* material,
        size_t num_of_shapes, size_t* next_triangle_id)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            cuda_shapes[i] = new Triangle(parent, (*next_triangle_id)++, material, shapes_data[i]);
            transformTriangle(object_to_world, world_to_object, cuda_shapes[i], &shapes_infos[i]);
        }
    }

    __global__ void transformTriangles(
        glm::mat4* object_to_world, glm::mat4* world_to_object,
        Shape** cuda_shapes, ShapeInfo* shapes_infos,
        size_t num_of_shapes)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            transformTriangle(object_to_world, world_to_object, cuda_shapes[i], &shapes_infos[i]);
        }
    }

    __global__ void changeTrianglesMaterial(
        Shape** cuda_shapes, size_t num_of_shapes,
        Material* material)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            cuda_shapes[i]->material = material;
        }
    }
    
}

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
      is_emitting_some_light(this->material->material->isEmittingLight()),
      num_of_shapes(this->model_data->triangles.size()),
      shapes(num_of_shapes),
      shapes_infos(num_of_shapes)
{
    createNameForObject();
    object_to_world.copyFrom(createObjectToWorldTransform());
    world_to_object.copyFrom(createWorldToObjectTransform());
    createMeshConsistingOfShapes();
}

void Object::createNameForObject()
{
    name = "object" + std::to_string(next_id++);
}

void Object::createMeshConsistingOfShapes()
{
    allocateShapesOnDeviceMemory();
    gatherShapesEmittingLight();
}

void Object::allocateShapesOnDeviceMemory()
{
    dmm::DeviceMemoryPointer<TriangleData> triangles_data(num_of_shapes);
    triangles_data.copyFrom(model_data->triangles.data());

    cudaObjectUtils::createTrianglesOnDeviceMemory<<<1, 1>>>(
        this,
        object_to_world.data(), world_to_object.data(),
        triangles_data.data(),
        shapes.data(), shapes_infos.data(),
        material->cuda_material.data(), num_of_shapes, next_shape_id.data());
    cudaDeviceSynchronize();
}

void Object::refreshMeshShapesTransforms()
{
    cudaObjectUtils::transformTriangles<<<1, 1>>>(object_to_world.data(), world_to_object.data(), shapes.data(), shapes_infos.data(), num_of_shapes);
    cudaDeviceSynchronize();
}

void Object::gatherShapesEmittingLight()
{
    shapes_emitting_light.clear();
    if (material->material->isEmittingLight())
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            for (const auto& vertex : model_data->triangles[i].vertices)
            {
                if (material->material->getSpecularValue(vertex.texture_coordinate).g > 0.5f)
                {
                    shapes_emitting_light.push_back(shapes[i]);
                }
            }
        }   
    }

    is_emitting_some_light = !shapes_emitting_light.empty();
}

void Object::changeMeshShapesMaterial()
{
    cudaObjectUtils::changeTrianglesMaterial<<<1, 1>>>(shapes.data(), num_of_shapes, material->cuda_material.data());
    cudaDeviceSynchronize();
}

size_t Object::getNumberOfShapes()
{
    return num_of_shapes;
}

std::vector<Shape*> Object::getShapesEmittingLight() const
{
    return shapes_emitting_light;
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
    object_to_world.copyFrom(createObjectToWorldTransform());
    world_to_object.copyFrom(createWorldToObjectTransform());
    refreshMeshShapesTransforms();
    parent_scene->notifyOnObjectChange();
}

void Object::changeMaterial(std::shared_ptr<MaterialAsset> new_material)
{
    material = std::move(new_material);
    changeMeshShapesMaterial();
    gatherShapesEmittingLight();
    parent_scene->notifyOnObjectMaterialChange(this);
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

Shape** Object::getShapes()
{
    return shapes.data();
}

ShapeInfo* Object::getShapesInfos()
{
    return shapes_infos.data();
}

ObjectInfo Object::getObjectInfo()
{
    return { name, model_data->model_name, material->material->name, position, rotation, scale };
}

bool Object::operator==(unsigned id) const
{
    return object_id == id;
}
