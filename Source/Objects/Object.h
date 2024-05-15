#pragma once

#include <glm/ext/matrix_transform.hpp>
#include "RenderEngine/Models/Model.h"
#include "Components/TransformComponent.h"
#include "Objects/Components/PointLightComponent.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/Models/BlasInstance.h"

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

    void setMaterial(std::shared_ptr<Material> in_material);
    std::shared_ptr<Material> getMaterial() { return material; }
    bool isLight() { return material->isLightMaterial(); };

    //TODO: create abstraction for creating blas instances
    void createBlasInstance();
    BlasInstance* getBlasInstance() { return &blas_instance; }
    //

    ObjectDescription getObjectDescription();

    TransformComponent transform_component;

private:
    explicit Object(id_t object_id) : id{object_id} {}

    id_t id;

    std::shared_ptr<Model> model;
    std::shared_ptr<Material> material;

    BlasInstance blas_instance{};
};
