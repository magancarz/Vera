#pragma once

#include <glm/ext/matrix_transform.hpp>
#include "RenderEngine/RenderingAPI/Model.h"
#include "Components/TransformComponent.h"
#include "Objects/Components/PointLightComponent.h"

struct PointLight
{
    glm::vec4 position{}; // ignore w
    glm::vec4 color{}; // w is intensity
};

class Object
{
public:
    using id_t = unsigned int;

    static Object createObject();
    static Object createPointLight(float intensity = 10.f, float radius = 0.1f, const glm::vec3& color = {1.f, 1.f, 1.f});

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    [[nodiscard]] id_t getID() const { return id; }

    glm::vec3 color{};
    TransformComponent transform_component;


    std::shared_ptr<Model> model;
    std::unique_ptr<PointLightComponent> point_light;

private:
    Object(id_t object_id) : id{object_id} {}

    id_t id;
};
