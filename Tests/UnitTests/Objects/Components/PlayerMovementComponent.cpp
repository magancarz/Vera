//#include "gtest/gtest.h"
//#include "Objects/Components/PlayerMovementComponent.h"
//
//#include "World/World.h"
//
//struct PlayerMovementComponentTests : public ::testing::Test
//{
//    Object owner;
//    World world;
//    std::shared_ptr<TransformComponent> transform_component;
//
//    void SetUp() override
//    {
//        owner = Object{};
//        world = World{};
//        transform_component = std::make_shared<TransformComponent>(&owner, &world);
//    }
//
//    void TearDown() override
//    {
//        transform_component.reset();
//    }
//};
//
//TEST_F(PlayerMovementComponentTests, shouldReturnIdentityTransformMatrixWhenUnchanged)
//{
//    // given
//    PlayerMovementComponent player_movement_component{&owner, &world, transform_component};
//
//    // when
//    glm::mat4 transform = transform_component.transform();
//
//    // then
//    EXPECT_EQ(transform, glm::mat4{1.f});
//}