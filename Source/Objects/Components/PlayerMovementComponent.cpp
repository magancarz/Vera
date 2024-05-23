#include "PlayerMovementComponent.h"

PlayerMovementComponent::PlayerMovementComponent(Object* owner, std::shared_ptr<InputManager> input_manager, std::shared_ptr<TransformComponent> transform_component)
    : ObjectComponent(owner), input_manager{std::move(input_manager)}, transform_component{std::move(transform_component)} {}

void PlayerMovementComponent::update(FrameInfo& frame_info)
{
    glm::vec3 rotate{0.f};
    if (input_manager->isKeyPressed(keys.look_right)) rotate.y += 1.f;
    if (input_manager->isKeyPressed(keys.look_left)) rotate.y -= 1.f;
    if (input_manager->isKeyPressed(keys.look_up)) rotate.x -= 1.f;
    if (input_manager->isKeyPressed(keys.look_down)) rotate.x += 1.f;

    if (glm::dot(rotate, rotate) > std::numeric_limits<float>::epsilon())
    {
        transform_component->rotation += look_speed * frame_info.frame_time * glm::normalize(rotate);
    }

    player_moved = rotate != glm::vec3{0};

    constexpr float RADIANS_85_DEGREES = glm::radians(85.f);
    transform_component->rotation.x = glm::clamp(transform_component->rotation.x, -RADIANS_85_DEGREES, RADIANS_85_DEGREES);
    transform_component->rotation.y = glm::mod(transform_component->rotation.y, glm::two_pi<float>());

    float yaw = transform_component->rotation.y;
    const glm::vec3 forward_dir{sin(yaw), 0.f, cos(yaw)};
    const glm::vec3 right_dir{forward_dir.z, 0.f, -forward_dir.x};
    const glm::vec3 up_dir{0.f, -1.f, 0.f};

    glm::vec3 move_dir{0.f};
    if (input_manager->isKeyPressed(keys.move_forward)) move_dir += forward_dir;
    if (input_manager->isKeyPressed(keys.move_backward)) move_dir -= forward_dir;
    if (input_manager->isKeyPressed(keys.move_right)) move_dir += right_dir;
    if (input_manager->isKeyPressed(keys.move_left)) move_dir -= right_dir;
    if (input_manager->isKeyPressed(keys.move_up)) move_dir += up_dir;
    if (input_manager->isKeyPressed(keys.move_down)) move_dir -= up_dir;

    player_moved |= move_dir != glm::vec3{0};

    if (glm::dot(move_dir, move_dir) > std::numeric_limits<float>::epsilon())
    {
        transform_component->translation += move_speed * frame_info.frame_time * glm::normalize(move_dir);
    }

    frame_info.need_to_refresh_generated_image |= player_moved;
}
