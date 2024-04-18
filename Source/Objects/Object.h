#pragma once

#include <glm/ext/matrix_transform.hpp>
#include "RenderEngine/RenderingAPI/Model.h"
#include "Components/TransformComponent.h"
#include "Objects/Components/PointLightComponent.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/Models/BlasInstance.h"

struct PointLight
{
    glm::vec4 position{}; // ignore w
    glm::vec4 color{}; // w is intensity
};

class Object
{
public:
    using id_t = uint32_t;

    static Object createObject();

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    [[nodiscard]] id_t getID() const { return id; }

    void setModel(std::shared_ptr<Model> in_model);
    std::shared_ptr<Model> getModel() { return model; }

    BlasInstance* getBlasInstance() { return &blas_instance; }
    ObjectDescription getObjectDescription() const;

    TransformComponent transform_component;

private:
    Object(id_t object_id) : id{object_id} {}

    id_t id;

    std::shared_ptr<Model> model;
    std::shared_ptr<Material> material;

    void createBlasInstance();

    BlasInstance blas_instance{};
};
