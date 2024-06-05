#pragma once

#include "imgui.h"

#include "GUIComponent.h"

class FramerateTextComponent : public GUIComponent
{
public:
    FramerateTextComponent();

    void update(FrameInfo& frame_info) override;

    inline static const char* const DISPLAY_NAME{"Framerate Text"};

private:
    ImGuiIO& io;
};
