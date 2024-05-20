#include <glm/ext/matrix_clip_space.hpp>
#include "gtest/gtest.h"
#include "Objects/Components/CameraComponent.h"

#include "World/World.h"
#include "TestUtils.h"

struct CameraComponentTests : public ::testing::Test
{
    Object owner;
    World world;
    std::shared_ptr<TransformComponent> transform_component;

    void SetUp() override
    {
        owner = Object{};
        world = World{};
        transform_component = std::make_shared<TransformComponent>(&owner, &world);
    }

    void TearDown() override
    {
        transform_component.reset();
    }
};

TEST_F(CameraComponentTests, shouldReturnCorrectProjectionMatrix)
{
    // given
    CameraComponent camera_component{&owner, &world, transform_component};
    const float fovy = 70.f, aspect = 1280.f / 800.f, near = 0.1f, far = 100.f;
    camera_component.setPerspectiveProjection(fovy, aspect, near, far);

    glm::mat4 expected_projection{0.f};
    const float tan_half_fov_y = tan(fovy / 2.f);
    expected_projection = glm::mat4{0.0f};
    expected_projection[0][0] = 1.f / (aspect * tan_half_fov_y);
    expected_projection[1][1] = -1.f / (tan_half_fov_y);
    expected_projection[2][2] = far / (far - near);
    expected_projection[2][3] = 1.f;
    expected_projection[3][2] = -(far * near) / (far - near);

    // when
    glm::mat4 projection = camera_component.getProjection();

    // then
    TestUtils::expectTwoMatricesToBeEqual(projection, expected_projection);
}

TEST_F(CameraComponentTests, shouldReturnCorrectViewMatrix)
{
    // given
    CameraComponent camera_component{&owner, &world, transform_component};
    const glm::vec3 camera_position{5.f, 10.f, -1.f};
    const glm::vec3 camera_rotation{glm::radians(55.f), glm::radians(90.f), 0};
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
    CameraComponent camera_component{&owner, &world, transform_component};
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
    CameraComponent camera_component{&owner, &world, transform_component};

    const glm::vec3 player_position{5.f, 10.f, -1.f};
    const glm::vec3 player_rotation{glm::radians(55.f), glm::radians(90.f), 0};
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
    CameraComponent camera_component{&owner, &world, transform_component};
    const float fovy = 70.f, aspect = 1280.f / 800.f, near = 0.1f, far = 100.f;
    camera_component.setPerspectiveProjection(fovy, aspect, near, far);

    glm::mat4 expected_projection{0.f};
    const float tan_half_fov_y = tan(fovy / 2.f);
    expected_projection = glm::mat4{0.0f};
    expected_projection[0][0] = 1.f / (aspect * tan_half_fov_y);
    expected_projection[1][1] = -1.f / (tan_half_fov_y);
    expected_projection[2][2] = far / (far - near);
    expected_projection[2][3] = 1.f;
    expected_projection[3][2] = -(far * near) / (far - near);

    FrameInfo frame_info{};

    // when
    camera_component.update(frame_info);

    // then
    TestUtils::expectTwoMatricesToBeEqual(frame_info.camera_projection_matrix, expected_projection);
}