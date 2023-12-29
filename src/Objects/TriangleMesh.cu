#include "TriangleMesh.h"
#include "helper_cuda.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <cuda_runtime.h>
#include <imgui.h>
#include <imgui_stdlib.h>

#include "Scene/Scene.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"
#include "Materials/MaterialAsset.h"
#include "Materials/Material.h"
#include "utils/DeviceObjectsAllocation.h"
#include "models/AssetManager.h"
#include "GUI/GUI.h"

namespace cudaObjectUtils
{
    __device__ void resetShapeTransform(Shape* shape)
    {
        shape->resetTransform();
    }

    __global__ void resetShapesTransforms(Shape** shape, size_t num_of_shapes)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            resetShapeTransform(shape[i]);
        }
    }

    __device__ void transformShape(
            glm::mat4* object_to_world, glm::mat4* world_to_object,
            Shape* shape, ShapeInfo* shapes_info)
    {
        shape->setTransform(object_to_world, world_to_object);
        shapes_info->world_bounds = shape->world_bounds;
    }

    __global__ void transformShapes(
            glm::mat4* object_to_world, glm::mat4* world_to_object,
            Shape** cuda_shapes, ShapeInfo* shapes_infos,
            size_t num_of_shapes)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            transformShape(object_to_world, world_to_object, cuda_shapes[i], &shapes_infos[i]);
        }
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
            transformShape(object_to_world, world_to_object, cuda_shapes[i], &shapes_infos[i]);
        }
    }

    __global__ void changeShapesMaterial(
            Shape** cuda_shapes, size_t num_of_shapes,
            Material* material)
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            cuda_shapes[i]->material = material;
        }
    }
}

TriangleMesh::TriangleMesh(
        Scene* parent_scene,
        std::shared_ptr<MaterialAsset> material,
        std::shared_ptr<RawModel> model_data,
        const glm::vec3& position,
        const glm::vec3& rotation,
        float scale)
        : Object(parent_scene, position),
          material(std::move(material)),
          model_data(std::move(model_data)),
          rotation(rotation),
          scale(scale),
          is_emitting_some_light(this->material->material->isEmittingLight()),
          num_of_shapes(this->model_data->triangles.size())
{
    createNameForObject();
    object_to_world.copyFrom(createObjectToWorldTransform());
    world_to_object.copyFrom(createWorldToObjectTransform());
}

TriangleMesh::~TriangleMesh()
{
    dmm::deleteObjectCUDA<Shape><<<1, 1>>>(shapes.data(), num_of_shapes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void TriangleMesh::createNameForObject()
{
    object_id = next_id++;
    name = "object" + std::to_string(object_id);
}

void TriangleMesh::createShapesForRayTracedMesh()
{
    createShapesOnDeviceMemory();
    gatherShapesEmittingLight();
}

void TriangleMesh::createShapesOnDeviceMemory()
{
    shapes = dmm::DeviceMemoryPointer<Shape*>(num_of_shapes);
    shapes_infos = dmm::DeviceMemoryPointer<ShapeInfo>(num_of_shapes);
    dmm::DeviceMemoryPointer<TriangleData> triangles_data(num_of_shapes);
    triangles_data.copyFrom(model_data->triangles.data());

    cudaObjectUtils::createTrianglesOnDeviceMemory<<<1, 1>>>(
        this,
        object_to_world.data(), world_to_object.data(),
        triangles_data.data(),
        shapes.data(), shapes_infos.data(),
        material->cuda_material.data(), num_of_shapes, next_shape_id.data());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void TriangleMesh::gatherShapesEmittingLight()
{
    shapes_emitting_light.clear();
    if (material->material->isEmittingLight())
    {
        for (size_t i = 0; i < num_of_shapes; ++i)
        {
            if (determineIfShapeIsEmittingLight(i))
            {
                shapes_emitting_light.push_back(shapes[i]);
            }
        }
    }

    is_emitting_some_light = !shapes_emitting_light.empty();
}

bool TriangleMesh::determineIfShapeIsEmittingLight(size_t i)
{
    for (const auto& vertex : model_data->triangles[i].vertices)
    {
        if (material->material->getSpecularValue(vertex.texture_coordinate).g > 0.5f)
        {
            return true;
        }
    }

    return false;
}

void TriangleMesh::changeMeshShapesMaterial()
{
    cudaObjectUtils::changeShapesMaterial<<<1, 1>>>(shapes.data(), num_of_shapes, material->cuda_material.data());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

size_t TriangleMesh::getNumberOfShapes()
{
    return num_of_shapes;
}

std::vector<Shape*> TriangleMesh::getShapesEmittingLight() const
{
    return shapes_emitting_light;
}

glm::mat4 TriangleMesh::createObjectToWorldTransform()
{
    glm::mat4 transformation_matrix = translate(glm::mat4(1.0f), position);
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.x), glm::vec3(1, 0, 0));
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.y), glm::vec3(0, 1, 0));
    transformation_matrix = rotate(transformation_matrix, glm::radians(rotation.z), glm::vec3(0, 0, 1));
    transformation_matrix = glm::scale(transformation_matrix, glm::vec3(scale));

    return transformation_matrix;
}

glm::mat4 TriangleMesh::createWorldToObjectTransform()
{
    glm::mat4 transformation_matrix = createObjectToWorldTransform();
    transformation_matrix = glm::inverse(transformation_matrix);

    return transformation_matrix;
}

void TriangleMesh::refreshObject()
{
    resetMeshTrianglesTransforms();
    object_to_world.copyFrom(createObjectToWorldTransform());
    world_to_object.copyFrom(createWorldToObjectTransform());
    transformMeshTriangles();
    parent_scene->notifyOnObjectChange();
}

void TriangleMesh::resetMeshTrianglesTransforms()
{
    cudaObjectUtils::resetShapesTransforms<<<1, 1>>>(shapes.data(), num_of_shapes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void TriangleMesh::transformMeshTriangles()
{
    cudaObjectUtils::transformShapes<<<1, 1>>>(object_to_world.data(), world_to_object.data(), shapes.data(), shapes_infos.data(), num_of_shapes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void TriangleMesh::changeMaterial(std::shared_ptr<MaterialAsset> new_material)
{
    material = std::move(new_material);
    changeMeshShapesMaterial();
    gatherShapesEmittingLight();
    parent_scene->notifyOnObjectMaterialChange();
}

void TriangleMesh::setPosition(const glm::vec3 &value)
{
    position = value;
    refreshObject();
}

glm::vec3 TriangleMesh::getRotation() const
{
    return rotation;
}

float TriangleMesh::getScale() const
{
    return scale;
}

void TriangleMesh::setRotation(const glm::vec3& value)
{
    rotation = value;
    refreshObject();
}

void TriangleMesh::setScale(float value)
{
    scale = value;
    refreshObject();
}

Material* TriangleMesh::getMaterial() const
{
    return material->cuda_material.data();
}

std::shared_ptr<RawModel> TriangleMesh::getModelData() const
{
    return model_data;
}

Shape** TriangleMesh::getShapes()
{
    return shapes.data();
}

ShapeInfo* TriangleMesh::getShapesInfos()
{
    return shapes_infos.data();
}

std::string TriangleMesh::getObjectInfo()
{
    TriangleMeshInfo object_info{name, model_data->model_name, material->material->name, position, rotation, scale };
    return object_info.toString();
}

void TriangleMesh::renderObjectInformationGUI()
{
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Object Details");
    ImGui::Text("Name");
    auto& object_name = name;
    ImGui::InputText("##name", &object_name);
    ImGui::Separator();

    const auto object_position = getPosition();
    float position[] =
    {
            object_position.x,
            object_position.y,
            object_position.z
    };
    ImGui::Text("Position");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * GUI::INPUT_FIELD_SIZE);
    ImGui::InputFloat3("##Position", position);
    if (ImGui::IsItemEdited())
    {
        setPosition({position[0], position[1], position[2]});
    }
    const auto object_rotation = getRotation();
    float rotation[] =
    {
            object_rotation.x,
            object_rotation.y,
            object_rotation.z
    };
    ImGui::Text("Rotation");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * GUI::INPUT_FIELD_SIZE);
    ImGui::InputFloat3("##Rotation", rotation);

    if (ImGui::IsItemEdited())
    {
        setRotation({rotation[0], rotation[1], rotation[2]});
    }

    float scale = getScale();
    ImGui::Text("Scale");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * GUI::INPUT_FIELD_SIZE / 3.f);
    ImGui::InputFloat("##Scale", &scale);

    if (ImGui::IsItemEdited())
    {
        setScale(scale);
    }

    ImGui::Separator();

    ImGui::Text("Material");
    const auto items = AssetManager::getAvailableMaterialAssets();
    const char* current_item_label = getMaterial()->name.c_str();
    if (ImGui::BeginCombo("##material", current_item_label))
    {
        for (auto& item : items)
        {
            const bool is_selected = (current_item_label == item->material->name);
            if (ImGui::Selectable(item->material->name.c_str(), is_selected))
            {
                if (ImGui::IsItemEdited())
                {
                    changeMaterial(item);
                }

                current_item_label = item->material->name.c_str();
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}