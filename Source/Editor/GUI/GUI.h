#pragma once

#include "RenderEngine/RenderingAPI/Device.h"
#include "RenderEngine/Window.h"
#include "RenderEngine/RenderingAPI/Descriptors.h"
#include "RenderEngine/RenderingAPI/SwapChain.h"
#include "RenderEngine/FrameInfo.h"

class GUI
{
public:
    GUI(Device& device, Window& window, std::shared_ptr<SwapChain> swap_chain);
    ~GUI();

    GUI(const GUI&) = delete;
    GUI& operator=(const GUI&) = delete;

    void updateGUIElements(FrameInfo& frame_info);
    void renderGUIElements(VkCommandBuffer command_buffer);

private:
    Device& device;
    Window& window;
    std::shared_ptr<SwapChain> swap_chain;

    void initializeImGui();
    void createContext();

    void createDescriptorPool();

    std::unique_ptr<DescriptorPool> descriptor_pool;

    void createRenderPass();

    VkRenderPass render_pass;

    void setupRendererBackends();

    void createFramebuffers();

    std::vector<VkFramebuffer> framebuffers;

    void startFrame();

    //TODO: move it to other component
    float previous_sun_yaw_angle{30.f};
    float sun_yaw_angle{30.f};

    float previous_sun_pitch_angle{30.f};
    float sun_pitch_angle{30.f};

    float previous_weather{0.05f};
    float weather{0.05f};
};