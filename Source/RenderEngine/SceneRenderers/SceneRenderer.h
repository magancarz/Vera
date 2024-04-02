#pragma once

#include "RenderEngine/FrameInfo.h"
#include "RenderEngine/SceneRenderers/Rasterized/Systems/SimpleRenderSystem.h"
#include "RenderEngine/SceneRenderers/Rasterized/Systems/PointLightSystem.h"
#include "RenderEngine/Materials/Material.h"

class SceneRenderer
{
public:
    virtual ~SceneRenderer() = default;

    virtual void renderScene(FrameInfo& frame_info) = 0;
};