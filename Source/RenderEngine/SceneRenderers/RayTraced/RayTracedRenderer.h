#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"
#include "World/World.h"

class RayTracedRenderer : public SceneRenderer
{
public:
    RayTracedRenderer(Device& device, World* world);

    void renderScene(FrameInfo& frame_info) override;

private:
    Device& device;
    World* world;

    RayTracingBuilder ray_tracing_builder;

    void queryRayTracingPipelineProperties();

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties;

    void createBottomLevelAccelerationStructure();
};
