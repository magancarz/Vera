#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"

#include "World/World.h"
#include "UnitTests/Mocks/MockObjectComponent.h"
#include "UnitTests/Mocks/MockWorld.h"
#include "TestUtils.h"

using ::testing::_;

struct WorldTests : public ::testing::Test
{
    Object object;

    void SetUp() override
    {
        object = Object{};
    }

    void TearDown() override {}
};

TEST_F(WorldTests, shouldRegisterComponent)
{
    // given
    World world{};
    auto transform_component = std::make_shared<TransformComponent>(&object);

    // when & then
    world.registerComponent(transform_component);
}

TEST_F(WorldTests, shouldUpdateRegisteredComponents)
{
    // given
    World world{};

    auto first_component = std::make_shared<MockObjectComponent>(&object);
    world.registerComponent(first_component);

    auto second_component = std::make_shared<MockObjectComponent>(&object);
    world.registerComponent(second_component);

    FrameInfo frame_info{};

    // then
    EXPECT_CALL(*first_component, update(_)).Times(1);
    EXPECT_CALL(*second_component, update(_)).Times(1);

    // when
    world.update(frame_info);
}

TEST_F(WorldTests, shouldRemoveDeprecatedComponentsBeforeUpdate)
{
    // given
    MockWorld world{};

    auto first_component = std::make_shared<MockObjectComponent>(&object);
    world.registerComponent(first_component);

    auto second_component = std::make_shared<MockObjectComponent>(&object);
    world.registerComponent(second_component);

    object = Object{};
    first_component.reset();
    second_component.reset();

    const int expected_number_of_components_left = 0;

    FrameInfo frame_info{};

    // when
    world.update(frame_info);

    // then
    EXPECT_EQ(world.getNumberOfRegisteredComponents(), expected_number_of_components_left);
}

TEST_F(WorldTests, shouldLoadProject)
{
    // given
    World world{};

    ProjectInfo project_info{};
    project_info.project_name = "dummy_project";
    project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy1"));
    project_info.objects_infos.emplace_back(TestUtils::createDummyObjectInfo("dummy2"));

    auto window = Window::get();
    InputManager::get(window->getGFLWwindow());
    VulkanFacade device{*window};
    auto asset_manager = AssetManager::get(&device);

    // when
    world.loadProject(project_info, asset_manager);

    // then
    EXPECT_EQ(world.rendered_objects.size(), project_info.objects_infos.size());
}