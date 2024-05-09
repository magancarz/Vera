#include "FramerateTextComponent.h"

FramerateTextComponent::FramerateTextComponent()
    : Component(DISPLAY_NAME), io{ImGui::GetIO()} {}

void FramerateTextComponent::update(FrameInfo& frame_info)
{
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
}