#include "gtest/gtest.h"

#include <glm/ext/matrix_clip_space.hpp>

#include "Objects/Components/CameraComponent.h"
#include "Objects/Components/TransformComponent.h"
#include "Objects/Object.h"
#include "RenderEngine/FrameInfo.h"

#include "TestUtils.h"

struct CameraComponentTests : public ::testing::Test
{
    Object owner;

    void SetUp() override
    {
        owner = Object{};
        owner.addRootComponent(std::make_unique<TransformComponent>(owner));
    }
};

TEST_F(CameraComponentTests, shouldReturnCorrectProjectionMatrix)
{
    // given
    CameraComponent camera_component{owner, owner.findComponentByClass<TransformComponent>()};
    constexpr float fovy = 70.f;
    constexpr float aspect = 1280.f / 800.f;
    camera_component.setPerspectiveProjection(fovy, aspect);

    glm::mat4 expected_projection;
    constexpr float tan_half_fov_y = glm::tan(fovy / 2.f);
    expected_projection = glm::mat4{0.0f};
    expected_projection[0][0] = 1.f / (aspect * tan_half_fov_y);
    expected_projection[1][1] = -1.f / tan_half_fov_y;
    expected_projection[2][2] = CameraComponent::CAMERA_FAR / (CameraComponent::CAMERA_FAR - CameraComponent::CAMERA_NEAR);
    expected_projection[2][3] = 1.f;
    expected_projection[3][2] = -(CameraComponent::CAMERA_FAR * CameraComponent::CAMERA_NEAR) / (CameraComponent::CAMERA_FAR - CameraComponent::CAMERA_NEAR);

    // when
    glm::mat4 projection = camera_component.getProjection();

    // then
    TestUtils::expectTwoMatricesToBeEqual(projection, expected_projection);
}

TEST_F(CameraComponentTests, shouldReturnCorrectViewMatrix)
{
    // given
    CameraComponent camera_component{owner, owner.findComponentByClass<TransformComponent>()};
    constexpr glm::vec3 camera_position{5.f, 10.f, -1.f};
    constexpr glm::vec3 camera_rotation{glm::radians(55.f), glm::radians(90.f), 0};
    camera_component.setViewYXZ(camera_position, camera_rotation);

    glm::mat4 rotation{1.f};
    rotation = glm::rotate(rotation, -camera_rotation.x, glm::vec3{1, 0, 0});
    rotation = glm::rotate(rotation, -camera_rotation.y, glm::vec3{0, 1, 0});
    rotation = glm::rotate(rotation, -camera_rotation.z, glm::vec3{0, 0, 1});

    glm::mat4 translation{1.f};
    translation = glm::translate(translation, -camera_position);

    const glm::mat4 expected_view = rotation * translation;

    // when
    glm::mat4 view = camera_component.getView();

    // then
    TestUtils::expectTwoMatricesToBeEqual(view, expected_view);
}

TEST_F(CameraComponentTests, shouldReturnCorrectInverseViewMatrix)
{
    // given
    CameraComponent camera_component{owner, owner.findComponentByClass<TransformComponent>()};
    const glm::vec3 camera_position{5.f, 10.f, -1.f};
    const glm::vec3 camera_rotation{glm::radians(55.f), glm::radians(90.f), 0};
    camera_component.setViewYXZ(camera_position, camera_rotation);

    glm::mat4 rotation{1.f};
    rotation = glm::rotate(rotation, -camera_rotation.x, glm::vec3{1, 0, 0});
    rotation = glm::rotate(rotation, -camera_rotation.y, glm::vec3{0, 1, 0});
    rotation = glm::rotate(rotation, -camera_rotation.z, glm::vec3{0, 0, 1});

    glm::mat4 translation{1.f};
    translation = glm::translate(translation, -camera_position);

    const glm::mat4 expected_inverse_view = glm::transpose(rotation * translation);

    // when
    glm::mat4 inverse_view = camera_component.getInverseView();

    // then
    TestUtils::expectTwoMatricesToBeEqual(inverse_view, expected_inverse_view);
}

TEST_F(CameraComponentTests, shouldUpdateViewMatrixInfoForEveryFrameBasedOfTransformComponentMovement)
{
    // given
    auto transform_component = owner.findComponentByClass<TransformComponent>();
    CameraComponent camera_component{owner, transform_component};

    constexpr glm::vec3 player_position{5.f, 10.f, -1.f};
    constexpr glm::vec3 player_rotation{glm::radians(55.f), glm::radians(90.f), 0};
    transform_component->translation = player_position;
    transform_component->rotation = player_rotation;

    glm::mat4 rotation{1.f};
    rotation = glm::rotate(rotation, -player_rotation.x, glm::vec3{1, 0, 0});
    rotation = glm::rotate(rotation, -player_rotation.y, glm::vec3{0, 1, 0});
    rotation = glm::rotate(rotation, -player_rotation.z, glm::vec3{0, 0, 1});

    glm::mat4 translation{1.f};
    translation = glm::translate(translation, -player_position);

    const glm::mat4 expected_view = rotation * translation;

    FrameInfo frame_info{};

    // when
    camera_component.update(frame_info);

    // then
    TestUtils::expectTwoMatricesToBeEqual(frame_info.camera_view_matrix, expected_view);
}

TEST_F(CameraComponentTests, shouldUpdateProjectionMatrixInfoForEveryFrame)
{
    // given
    CameraComponent camera_component{owner, owner.findComponentByClass<TransformComponent>()};
    constexpr float fovy = 70.f;
    constexpr float aspect = 1280.f / 800.f;
    camera_component.setPerspectiveProjection(fovy, aspect);

    glm::mat4 expected_projection;
    constexpr float tan_half_fov_y = glm::tan(fovy / 2.f);
    expected_projection = glm::mat4{0.0f};
    expected_projection[0][0] = 1.f / (aspect * tan_half_fov_y);
    expected_projection[1][1] = -1.f / tan_half_fov_y;
    expected_projection[2][2] = CameraComponent::CAMERA_FAR / (CameraComponent::CAMERA_FAR - CameraComponent::CAMERA_NEAR);
    expected_projection[2][3] = 1.f;
    expected_projection[3][2] = -(CameraComponent::CAMERA_FAR * CameraComponent::CAMERA_NEAR) / (CameraComponent::CAMERA_FAR - CameraComponent::CAMERA_NEAR);

    FrameInfo frame_info{};

    // when
    camera_component.update(frame_info);

    // then
    TestUtils::expectTwoMatricesToBeEqual(frame_info.camera_projection_matrix, expected_projection);
}

TEST_F(CameraComponentTests, shouldSkipUpdateAndFetchTransformComponentIfReferenceIsMissing)
{
    // given
    auto transform_component = owner.findComponentByClass<TransformComponent>();
    transform_component->translation = {5.f, 10.f, -1.f};

    glm::mat4 expected_second_frame_view_transform{1.f};
    expected_second_frame_view_transform = glm::translate(expected_second_frame_view_transform, transform_component->translation);

    CameraComponent camera_component{owner, nullptr};

    FrameInfo first_frame_info{};
    FrameInfo second_frame_info{};

    // when (should fetch transform component and skip update)
    camera_component.update(first_frame_info);

    // then
    TestUtils::expectTwoMatricesToBeEqual(first_frame_info.camera_view_matrix, glm::mat4{1.f});

    // when (should generate correct view matrix)
    camera_component.update(second_frame_info);

    // then
    TestUtils::expectTwoMatricesToBeEqual(second_frame_info.camera_view_matrix, expected_second_frame_view_transform);
}