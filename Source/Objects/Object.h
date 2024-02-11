#pragma once

#include <glm/ext/matrix_transform.hpp>
#include "RenderEngine/RenderingAPI/Model.h"
#include "Components/TransformComponent.h"

class Object
{
public:
    using id_t = unsigned int;

    static Object createObject()
    {
        static id_t available_id = 0;
        return Object{available_id++};
    }

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    [[nodiscard]] id_t getID() const { return id; }

    std::shared_ptr<Model> model;
    glm::vec3 color{};
    TransformComponent transform_component;

private:
    Object(id_t object_id) : id{object_id} {}

    id_t id;
};
