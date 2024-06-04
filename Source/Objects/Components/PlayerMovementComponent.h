#pragma once

#include <memory>

#include "ObjectComponent.h"

class TransformComponent;
class InputManager;

class PlayerMovementComponent : public ObjectComponent
{
public:
    PlayerMovementComponent(Object& owner, InputManager& input_manager, std::shared_ptr<TransformComponent> transform_component);

    void update(FrameInfo& frame_info) override;

    [[nodiscard]] float getMoveSpeed() const { return move_speed; }
    [[nodiscard]] float getLookSpeed() const { return look_speed; }

private:
    InputManager& input_manager;

    std::shared_ptr<TransformComponent> transform_component;

    float move_speed{12.f};
    float look_speed{2.f};

    bool player_moved{false};
};
