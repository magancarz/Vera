#pragma once

#include "RenderEngine/FrameInfo.h"

class SceneRenderer
{
public:
    virtual ~SceneRenderer() = default;

    virtual void renderScene(FrameInfo& frame_info) = 0;
};