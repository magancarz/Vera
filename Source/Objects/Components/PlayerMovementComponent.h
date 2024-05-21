#pragma once

#include "Objects/Object.h"
#include "RenderEngine/Window.h"
#include "Input/InputManager.h"
#include "ObjectComponent.h"
#include "TransformComponent.h"

class PlayerMovementComponent : public ObjectComponent
{
public:
    struct KeyMappings
    {
        int move_left = GLFW_KEY_A;
        int move_right = GLFW_KEY_D;
        int move_forward = GLFW_KEY_W;
        int move_backward = GLFW_KEY_S;
        int move_up = GLFW_KEY_E;
        int move_down = GLFW_KEY_Q;
        int look_left = GLFW_KEY_LEFT;
        int look_right = GLFW_KEY_RIGHT;
        int look_up = GLFW_KEY_UP;
        int look_down = GLFW_KEY_DOWN;
    };

    explicit PlayerMovementComponent(Object* owner, std::shared_ptr<TransformComponent> transform_component);

    void update(FrameInfo& frame_info) override;
    [[nodiscard]] bool playerMoved() const;

private:
    KeyMappings keys{};
    std::shared_ptr<InputManager> input_manager;

    std::shared_ptr<TransformComponent> transform_component;

    float move_speed{12.f};
    float look_speed{2.f};

    bool player_moved{false};
};
