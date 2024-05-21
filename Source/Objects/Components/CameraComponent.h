#pragma once

#include "ObjectComponent.h"
#include "TransformComponent.h"

class CameraComponent : public ObjectComponent
{
public:
    explicit CameraComponent(Object* owner, std::shared_ptr<TransformComponent> transform_component);

    void update(FrameInfo& frame_info) override;

    void setPerspectiveProjection(float fovy, float aspect, float near, float far);
    void setViewYXZ(glm::vec3 position, glm::vec3 rotation);

    [[nodiscard]] const glm::mat4& getProjection() const { return projection; }
    [[nodiscard]] const glm::mat4& getView() const { return view; }
    [[nodiscard]] const glm::mat4& getInverseView() const { return inverse_view; }

private:
    std::shared_ptr<TransformComponent> transform_component;

    glm::mat4 projection{1.f};
    glm::mat4 view{1.f};
    glm::mat4 inverse_view{1.f};
};
