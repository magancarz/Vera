#include <TestUtils.h>
#include <Assets/AssetManager.h>
#include <Editor/Window/WindowSystem.h>
#include <Mocks/MockInputManager.h>
#include <RenderEngine/FrameInfo.h>
#include <RenderEngine/Renderer.h>
#include <World/World.h>

#include "gtest/gtest.h"

TEST(E2ETests, shouldMakeCorrectApplicationRun)
{
    // given
    auto asset_manager = std::make_unique<AssetManager>(TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator());
    auto input_manager = std::make_unique<MockInputManager>();

    World world{};
    ProjectInfo project_info = ProjectUtils::loadProject("debug");
    world.loadProject(project_info, *asset_manager);
    world.createViewerObject(*input_manager);

    auto renderer = std::make_unique<Renderer>(WindowSystem::get(), TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), world, *asset_manager);

    constexpr float FPS = 30.f;

    // when
    for (size_t i = 0; i < 30; ++i)
    {
        FrameInfo frame_info{};

        frame_info.delta_time = 1.0f / FPS;

        world.update(frame_info);
        renderer->render(frame_info);
    }

    // then
    vkDeviceWaitIdle(TestsEnvironment::vulkanHandler().getDeviceHandle());
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}