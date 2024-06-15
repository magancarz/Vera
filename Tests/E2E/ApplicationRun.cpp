#include <TestUtils.h>
#include <Assets/AssetManager.h>
#include <Editor/Window/GLFWWindow.h>
#include <Editor/Window/WindowSystem.h>
#include <Mocks/MockInputManager.h>
#include <RenderEngine/FrameInfo.h>
#include <RenderEngine/Renderer.h>
#include <World/World.h>

#include "gtest/gtest.h"

struct ApplicationTests : public ::testing::Test
{
    inline static std::unique_ptr<AssetManager> asset_manager;
    inline static std::unique_ptr<MockInputManager> input_manager;
    inline static std::unique_ptr<Renderer> renderer;
    inline static World world{};

    static constexpr float FPS = 30.f;

    static void SetUpTestSuite()
    {
        asset_manager = std::make_unique<AssetManager>(TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator());
        input_manager = std::make_unique<MockInputManager>();

        ProjectInfo project_info = ProjectUtils::loadProject("debug");
        world.loadProject(project_info, *asset_manager);
        world.createViewerObject(*input_manager);

        renderer = std::make_unique<Renderer>(WindowSystem::get(), TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), world, *asset_manager);
    }

    static void TearDownTestSuite()
    {
        renderer.reset();
        world = World{};
        input_manager.reset();
        asset_manager.reset();
    }

    static void runFrame()
    {
        FrameInfo frame_info{};

        frame_info.delta_time = 1.0f / FPS;

        world.update(frame_info);
        renderer->render(frame_info);
    }

    static void runFrames(uint32_t num_of_frames)
    {
        for (uint32_t i = 0; i < num_of_frames; ++i)
        {
            runFrame();
        }
    }
};

TEST_F(ApplicationTests, shouldMakeCorrectApplicationRun)
{
    // when
    runFrames(30);

    // then
    vkDeviceWaitIdle(TestsEnvironment::vulkanHandler().getDeviceHandle());
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}

TEST_F(ApplicationTests, shouldHandleWindowResizing)
{
    // given
    runFrames(5);

    GLFWwindow* glfw_window = dynamic_cast<GLFWWindow*>(&WindowSystem::get())->getGFLWwindow();
    glfwSetWindowSize(glfw_window, Editor::DEFAULT_WINDOW_WIDTH / 2, Editor::DEFAULT_WINDOW_HEIGHT / 2);

    // when
    runFrames(5);

    // then
    glfwSetWindowSize(glfw_window, Editor::DEFAULT_WINDOW_WIDTH, Editor::DEFAULT_WINDOW_HEIGHT);
    runFrames(5);

    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
    vkDeviceWaitIdle(TestsEnvironment::vulkanHandler().getDeviceHandle());
}
