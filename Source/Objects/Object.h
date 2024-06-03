#pragma once

#include <memory>

#include <glm/glm.hpp>

class ObjectComponent;
class TransformComponent;

class Object
{
public:
    using id_t = uint32_t;

    Object();

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    [[nodiscard]] id_t getID() const { return id; }

    glm::vec3 getLocation();
    glm::mat4 getTransform();

    void addComponent(std::shared_ptr<ObjectComponent> component);
    void addRootComponent(std::shared_ptr<TransformComponent> transform_component);

    template <typename T>
    T* findComponentByClass()
    {
        for (auto& component : components)
        {
            if (auto matching_class_component = dynamic_cast<T*>(component.get()))
            {
                return matching_class_component;
            }
        }

        return nullptr;
    }

private:
    explicit Object(id_t object_id) : id{object_id} {}

    inline static id_t available_id = 0;
    id_t id;

    std::vector<std::shared_ptr<ObjectComponent>> components;
    TransformComponent* root_component{nullptr};
};
