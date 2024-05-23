#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"

#include "Objects/Components/PlayerMovementComponent.h"

#include "UnitTests/Mocks/MockInputManager.h"
#include "TestUtils.h"

using testing::Return;
using testing::AtLeast;
using testing::_;

struct PlayerMovementComponentTests : public ::testing::Test
{
    Object owner;
    std::shared_ptr<TransformComponent> transform_component;

    void SetUp() override
    {
        owner = Object{};
        transform_component = std::make_shared<TransformComponent>(&owner);
    }

    void TearDown() override
    {
        transform_component.reset();
    }
};

TEST_F(PlayerMovementComponentTests, shouldCheckMovementOnEveryAxis)
{
    // given
    auto mock_input_manager = std::make_shared<MockInputManager>();
    PlayerMovementComponent player_movement_component{&owner, mock_input_manager, transform_component};
    PlayerMovementComponent::KeyMappings key_mappings = player_movement_component.getKeyMappings();

    // then
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_left)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_right)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_forward)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_backward)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_up)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_down)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_left)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_right)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_up)).Times(1);
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_down)).Times(1);

    FrameInfo frame_info{};
    float delta_time = 1.f;
    frame_info.frame_time = delta_time;

    // when
    player_movement_component.update(frame_info);
}

TEST_F(PlayerMovementComponentTests, shouldApplyCorrectTranslationWithSomeKeysPressed)
{
    // given
    auto mock_input_manager = std::make_shared<MockInputManager>();
    PlayerMovementComponent player_movement_component{&owner, mock_input_manager, transform_component};
    PlayerMovementComponent::KeyMappings key_mappings = player_movement_component.getKeyMappings();

    FrameInfo frame_info{};
    float delta_time = 1.f;
    frame_info.frame_time = delta_time;

    glm::vec3 expected_move_dir{-1, 0, -1};
    glm::vec3 expected_translation
    {
        player_movement_component.getMoveSpeed() * delta_time,
        0,
        player_movement_component.getMoveSpeed() * delta_time
    };
    expected_translation *= glm::normalize(expected_move_dir);

    EXPECT_CALL(*mock_input_manager, isKeyPressed(_)).Times(AtLeast(1));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_backward)).WillOnce(Return(true));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.move_left)).WillOnce(Return(true));

    // when
    player_movement_component.update(frame_info);

    // then
    TestUtils::expectTwoVectorsToBeEqual(transform_component->translation, expected_translation);
}

TEST_F(PlayerMovementComponentTests, shouldApplyCorrectRotationWithSomeKeysPressed)
{
    // given
    auto mock_input_manager = std::make_shared<MockInputManager>();
    PlayerMovementComponent player_movement_component{&owner, mock_input_manager, transform_component};
    PlayerMovementComponent::KeyMappings key_mappings = player_movement_component.getKeyMappings();

    FrameInfo frame_info{};
    float delta_time = .1f;
    frame_info.frame_time = delta_time;

    glm::vec3 expected_look_dir{1, 1, 0};
    glm::vec3 expected_rotation
    {
            player_movement_component.getLookSpeed() * delta_time,
            player_movement_component.getLookSpeed() * delta_time,
            0,
    };
    expected_rotation *= glm::normalize(expected_look_dir);

    EXPECT_CALL(*mock_input_manager, isKeyPressed(_)).Times(AtLeast(1));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_down)).WillOnce(Return(true));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_right)).WillOnce(Return(true));

    // when
    player_movement_component.update(frame_info);

    // then
    TestUtils::printVector(transform_component->rotation);
    TestUtils::printVector(expected_rotation);
    TestUtils::expectTwoVectorsToBeEqual(transform_component->rotation, expected_rotation);
}

TEST_F(PlayerMovementComponentTests, shouldClampRotationWhenValuesAreTooLarge)
{
    // given
    auto mock_input_manager = std::make_shared<MockInputManager>();
    PlayerMovementComponent player_movement_component{&owner, mock_input_manager, transform_component};
    PlayerMovementComponent::KeyMappings key_mappings = player_movement_component.getKeyMappings();

    FrameInfo frame_info{};
    float delta_time = 5.f;
    frame_info.frame_time = delta_time;

    constexpr float RADIANS_85_DEGREES = glm::radians(85.f);
    glm::vec3 expected_look_dir{1, 1, 0};
    glm::vec3 initial_rotation
    {
            player_movement_component.getLookSpeed() * delta_time,
            player_movement_component.getLookSpeed() * delta_time,
            0,
    };
    initial_rotation *= glm::normalize(expected_look_dir);
    glm::vec3 expected_rotation = initial_rotation;
    expected_rotation.x = glm::clamp(expected_rotation.x, -RADIANS_85_DEGREES, RADIANS_85_DEGREES);
    expected_rotation.y = glm::mod(expected_rotation.y, glm::two_pi<float>());

    EXPECT_CALL(*mock_input_manager, isKeyPressed(_)).Times(AtLeast(1));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_down)).WillOnce(Return(true));
    EXPECT_CALL(*mock_input_manager, isKeyPressed(key_mappings.look_right)).WillOnce(Return(true));

    // when
    player_movement_component.update(frame_info);

    // then
    TestUtils::printVector(transform_component->rotation);
    TestUtils::printVector(expected_rotation);
    TestUtils::expectTwoVectorsToBeEqual(transform_component->rotation, expected_rotation);
}