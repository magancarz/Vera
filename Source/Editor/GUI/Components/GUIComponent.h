#pragma once

#include "RenderEngine/FrameInfo.h"

class GUIComponent
{
public:
    explicit GUIComponent(std::string name);

    virtual ~GUIComponent() = default;

    virtual void update(FrameInfo& frame_info) = 0;

    std::string getName() const { return component_name; }

protected:
    std::string component_name;
};