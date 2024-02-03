#pragma once

#include "RenderEngine/RenderingAPI/Model.h"

struct Transform2DComponent
{
    glm::vec2 translation{};
    glm::vec2 scale{1.f};

    glm::mat2 mat2() { return glm::mat2{{scale.x, 0.f}, {0.f, scale.y}}; }
};

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
    Transform2DComponent transform_2d;

private:
    Object(id_t object_id) : id{object_id} {}

    id_t id;
};
