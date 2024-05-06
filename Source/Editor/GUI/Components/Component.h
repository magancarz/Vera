#pragma once

#include "RenderEngine/FrameInfo.h"

class Component
{
public:
    explicit Component(std::string name);

    virtual ~Component() = default;

    virtual void update(FrameInfo& frame_info) = 0;

    std::string getName() { return component_name; }

protected:
    std::string component_name;
};