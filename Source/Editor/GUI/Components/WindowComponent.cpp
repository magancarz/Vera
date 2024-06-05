#include "WindowComponent.h"
#include "imgui.h"

WindowComponent::WindowComponent(std::string window_name)
    : GUIContainer(std::move(window_name)) {}

void WindowComponent::update(FrameInfo& frame_info)
{
    ImGui::Begin(component_name.c_str());

    GUIContainer::update(frame_info);

    ImGui::End();
}