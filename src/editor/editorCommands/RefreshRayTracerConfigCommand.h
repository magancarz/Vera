#pragma once

#include "EditorCommand.h"
#include "../../RenderEngine/RayTracing/RayTracerConfig.h"

class RefreshRayTracerConfigCommand : public EditorCommand
{
public:
    RefreshRayTracerConfigCommand(RayTracerConfig config);

    void execute(Editor* editor) override;

private:
    RayTracerConfig config;
};