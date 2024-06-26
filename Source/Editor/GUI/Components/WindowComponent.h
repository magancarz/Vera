#pragma once

#include "GUIContainer.h"

class WindowComponent : public GUIContainer
{
public:
    WindowComponent(std::string window_name);

    void update(FrameInfo& frame_info) override;
};
