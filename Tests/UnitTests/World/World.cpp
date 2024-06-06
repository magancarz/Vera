// #include "gtest/gtest.h"
// #include "gmock/gmock-matchers.h"
//
// #include "World/World.h"
// #include "Objects/Components/TransformComponent.h"
//
// #include "Objects/Components/CameraComponent.h"
// #include "UnitTests/Mocks/MockObjectComponent.h"
// #include "UnitTests/Mocks/MockWorld.h"
// #include "UnitTests/Mocks/MockInputManager.h"
// #include "UnitTests/Mocks/MockMemoryAllocator.h"
// #include "TestUtils.h"
//
// using ::testing::_;
//
// struct WorldTests : public ::testing::Test
// {
//     Object object;
//     std::unique_ptr<MemoryAllocator> memory_allocator = std::make_unique<MockMemoryAllocator>();
//
//     void SetUp() override
//     {
//         object = Object{};
//     }
//
//     void TearDown() override {}
// };
//
// TEST_F(WorldTests, shouldRegisterComponent)
// {
//     // given
//     World world{};
//     auto transform_component = std::make_shared<TransformComponent>(&object);
//
//     // when & then
//     world.registerComponent(transform_component);
// }
//
// TEST_F(WorldTests, shouldUpdateRegisteredComponents)
// {
//     // given
//     World world{};
//
//     auto first_component = std::make_shared<MockObjectComponent>(&object);
//     world.registerComponent(first_component);
//
//     auto second_component = std::make_shared<MockObjectComponent>(&object);
//     world.registerComponent(second_component);
//
//     FrameInfo frame_info{};
//
//     // then
//     EXPECT_CALL(*first_component, update(_)).Times(1);
//     EXPECT_CALL(*second_component, update(_)).Times(1);
//
//     // when
//     world.update(frame_info);
// }
//
// TEST_F(WorldTests, shouldRemoveDeprecatedComponentsBeforeUpdate)
// {
//     // given
//     MockWorld world{};
//
//     auto first_component = std::make_shared<MockObjectComponent>(&object);
//     world.registerComponent(first_component);
//
//     auto second_component = std::make_shared<MockObjectComponent>(&object);
//     world.registerComponent(second_component);
//
//     object = Object{};
//     first_component.reset();
//     second_component.reset();
//
//     const int expected_number_of_components_left = 0;
//
//     FrameInfo frame_info{};
//
//     // when
//     world.update(frame_info);
//
//     // then
//     EXPECT_EQ(world.getNumberOfRegisteredComponents(), expected_number_of_components_left);
// }
//
// TEST_F(WorldTests, shouldLoadObjectsFromProjectInfo)
// {
//     // given
//     World world{};
//
//     ProjectInfo project_info{};
//     project_info.project_name = "dummy_project";
//     project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy1"));
//     project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy2"));
//
//     auto mock_asset_manager = std::make_shared<MockAssetManager>(memory_allocator);
//
//     // when
//     world.loadProject(project_info, mock_asset_manager);
//
//     // then
//     EXPECT_EQ(world.getObjects().size(), project_info.objects_infos.size());
// }
//
// TEST_F(WorldTests, shouldCreateViewerObject)
// {
//     // given
//     World world{};
//
//     auto mock_input_manager = std::make_shared<MockInputManager>();
//
//     // when
//     world.createViewerObject(mock_input_manager);
//
//     // then
//     auto viewer_object = world.getViewerObject();
//     EXPECT_TRUE(viewer_object->findComponentByClass<TransformComponent>() != nullptr);
//     EXPECT_TRUE(viewer_object->findComponentByClass<PlayerMovementComponent>() != nullptr);
//     EXPECT_TRUE(viewer_object->findComponentByClass<CameraComponent>() != nullptr);
// }