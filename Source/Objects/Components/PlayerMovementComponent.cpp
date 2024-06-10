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

    const glm::vec3 player_frame_rotation = rotatePlayerWithInput(frame_info);
    const glm::vec3 player_frame_translation = translatePlayerWithInput(frame_info);
    frame_info.need_to_refresh_generated_image |= checkIfPlayerHasMoved(player_frame_rotation, player_frame_translation);
}

glm::vec3 PlayerMovementComponent::rotatePlayerWithInput(const FrameInfo& frame_info) const
{
    glm::vec3 rotation_input{0.0f};
    rotation_input.y += 1.0f * static_cast<float>(input_manager.isKeyPressed(LOOK_RIGHT));
    rotation_input.y -= 1.0f * static_cast<float>(input_manager.isKeyPressed(LOOK_LEFT));
    rotation_input.x -= 1.0f * static_cast<float>(input_manager.isKeyPressed(LOOK_UP));
    rotation_input.x += 1.0f * static_cast<float>(input_manager.isKeyPressed(LOOK_DOWN));

    if (glm::dot(rotation_input, rotation_input) > std::numeric_limits<float>::epsilon())
    {
        transform_component->rotation += look_speed * frame_info.delta_time * glm::normalize(rotation_input);
    }

    constexpr float X_ROTATION_LIMIT = glm::radians(85.f);
    transform_component->rotation.x = glm::clamp(transform_component->rotation.x, -X_ROTATION_LIMIT, X_ROTATION_LIMIT);
    transform_component->rotation.y = glm::mod(transform_component->rotation.y, glm::two_pi<float>());

    return rotation_input;
}

glm::vec3 PlayerMovementComponent::translatePlayerWithInput(const FrameInfo& frame_info) const
{
    const float yaw = transform_component->rotation.y;
    const glm::vec3 forward_dir{glm::sin(yaw), 0.f, glm::cos(yaw)};
    const glm::vec3 right_dir{forward_dir.z, 0.f, -forward_dir.x};
    constexpr glm::vec3 up_dir{0.f, -1.f, 0.f};

    glm::vec3 translation_input{0.f};
    translation_input += forward_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_FORWARD));
    translation_input -= forward_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_BACKWARD));
    translation_input += right_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_RIGHT));
    translation_input -= right_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_LEFT));
    translation_input += up_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_UP));
    translation_input -= up_dir * static_cast<float>(input_manager.isKeyPressed(MOVE_DOWN));

    if (glm::dot(translation_input, translation_input) > std::numeric_limits<float>::epsilon())
    {
        transform_component->translation += move_speed * frame_info.delta_time * glm::normalize(translation_input);
    }

    return translation_input;
}

bool PlayerMovementComponent::checkIfPlayerHasMoved(const glm::vec3& player_frame_rotation, const glm::vec3& player_frame_translation) const
{
    return player_frame_translation != glm::vec3{0} || player_frame_rotation != glm::vec3{0};
}
