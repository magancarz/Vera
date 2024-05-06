#pragma once

#include "Container.h"

class WindowComponent : public Container
{
public:
    WindowComponent(std::string window_name);

    void update(FrameInfo& frame_info) override;
};
