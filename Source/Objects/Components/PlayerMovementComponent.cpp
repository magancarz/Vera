#include "PlayerMovementComponent.h"

#include "RenderEngine/FrameInfo.h"
#include "Input/InputManager.h"
#include "TransformComponent.h"
#include "Input/KeyCodes.h"
#include "Logs/LogSystem.h"
#include "Objects/Object.h"

PlayerMovementComponent::PlayerMovementComponent(Object& owner, InputManager& input_manager, TransformComponent* transform_component)
    : ObjectComponent(owner), input_manager{input_manager}, transform_component{transform_component} {}

void PlayerMovementComponent::update(FrameInfo& frame_info)
{
    if (!transform_component)
    {
        LogSystem::log(LogSeverity::ERROR, "Missing transform component. Skipping frame and trying to fetch it from owner components...");
        transform_component = owner.findComponentByClass<TransformComponent>();
        return;
    }

    glm::vec3 rotate{0.f};
    if (input_manager.isKeyPressed(LOOK_RIGHT)) rotate.y += 1.f;
    if (input_manager.isKeyPressed(LOOK_LEFT)) rotate.y -= 1.f;
    if (input_manager.isKeyPressed(LOOK_UP)) rotate.x -= 1.f;
    if (input_manager.isKeyPressed(LOOK_DOWN)) rotate.x += 1.f;

    if (glm::dot(rotate, rotate) > std::numeric_limits<float>::epsilon())
    {
        transform_component->rotation += look_speed * frame_info.delta_time * glm::normalize(rotate);
    }

    player_moved = rotate != glm::vec3{0};

    constexpr float RADIANS_85_DEGREES = glm::radians(85.f);
    transform_component->rotation.x = glm::clamp(transform_component->rotation.x, -RADIANS_85_DEGREES, RADIANS_85_DEGREES);
    transform_component->rotation.y = glm::mod(transform_component->rotation.y, glm::two_pi<float>());

    float yaw = transform_component->rotation.y;
    const glm::vec3 forward_dir{glm::sin(yaw), 0.f, glm::cos(yaw)};
    const glm::vec3 right_dir{forward_dir.z, 0.f, -forward_dir.x};
    const glm::vec3 up_dir{0.f, -1.f, 0.f};

    glm::vec3 move_dir{0.f};
    if (input_manager.isKeyPressed(MOVE_FORWARD)) move_dir += forward_dir;
    if (input_manager.isKeyPressed(MOVE_BACKWARD)) move_dir -= forward_dir;
    if (input_manager.isKeyPressed(MOVE_RIGHT)) move_dir += right_dir;
    if (input_manager.isKeyPressed(MOVE_LEFT)) move_dir -= right_dir;
    if (input_manager.isKeyPressed(MOVE_UP)) move_dir += up_dir;
    if (input_manager.isKeyPressed(MOVE_DOWN)) move_dir -= up_dir;

    player_moved |= move_dir != glm::vec3{0};

    if (glm::dot(move_dir, move_dir) > std::numeric_limits<float>::epsilon())
    {
        transform_component->translation += move_speed * frame_info.delta_time * glm::normalize(move_dir);
    }

    frame_info.need_to_refresh_generated_image |= player_moved;
}
