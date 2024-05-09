#pragma once

#include "imgui.h"

#include "Component.h"

class FramerateTextComponent : public Component
{
public:
    FramerateTextComponent();

    void update(FrameInfo& frame_info) override;

    inline static const char* const DISPLAY_NAME{"Framerate Text"};

private:
    ImGuiIO& io;
};
