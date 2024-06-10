#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"

#include "World/World.h"

#include <Environment.h>

#include "Objects/Components/TransformComponent.h"

#include "Objects/Components/CameraComponent.h"
#include "Mocks/MockInputManager.h"
#include "Mocks/MockObject.h"
#include "TestUtils.h"

using ::testing::_;

TEST(WorldTests, shouldUpdateEveryWorldObject)
{
    // given
    World world;
    auto first_object = world.addObject<MockObject>(std::make_unique<MockObject>());
    auto second_object = world.addObject<MockObject>(std::make_unique<MockObject>());

    FrameInfo frame_info{};

    // then
    EXPECT_CALL(*first_object, update(_)).Times(1);
    EXPECT_CALL(*second_object, update(_)).Times(1);

    // when
    world.update(frame_info);
}

TEST(WorldTests, shouldLoadObjectsFromProjectInfo)
{
    // given
    World world{};

    ProjectInfo project_info{};
    project_info.project_name = "dummy_project";
    project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy1"));
    project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy2"));

    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};

    // when
    world.loadProject(project_info, asset_manager);

    // then
    EXPECT_EQ(world.getObjects().size(), project_info.objects_infos.size());
}

TEST(WorldTests, shouldCreateViewerObject)
{
    // given
    World world{};

    auto mock_input_manager = std::make_shared<MockInputManager>();

    // when
    world.createViewerObject(*mock_input_manager);

    // then
    auto viewer_object = world.getViewerObject();
    EXPECT_TRUE(viewer_object->findComponentByClass<TransformComponent>() != nullptr);
    EXPECT_TRUE(viewer_object->findComponentByClass<PlayerMovementComponent>() != nullptr);
    EXPECT_TRUE(viewer_object->findComponentByClass<CameraComponent>() != nullptr);
}