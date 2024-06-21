#pragma once

#include <memory>
#include <glm/vec3.hpp>

#include "ObjectComponent.h"

class TransformComponent;
class InputManager;

class PlayerMovementComponent : public ObjectComponent
{
public:
    PlayerMovementComponent(Object& owner, InputManager& input_manager, TransformComponent* transform_component);

    void update(FrameInfo& frame_info) override;

    [[nodiscard]] float getMoveSpeed() const { return move_speed; }
    [[nodiscard]] float getLookSpeed() const { return look_speed; }

private:
    InputManager& input_manager;

    TransformComponent* transform_component;

    [[nodiscard]] glm::vec3 rotatePlayerWithInput(const FrameInfo& frame_info) const;

    float look_speed{2.f};

    [[nodiscard]] glm::vec3 translatePlayerWithInput(const FrameInfo& frame_info) const;

    float move_speed{12.f};

    [[nodiscard]] bool checkIfPlayerHasMoved(const glm::vec3& player_frame_rotation, const glm::vec3& player_frame_translation) const;

    bool player_moved{false};
};
