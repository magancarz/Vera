#include "RefreshRayTracerConfigCommand.h"

#include "Editor/Editor.h"

RefreshRayTracerConfigCommand::RefreshRayTracerConfigCommand(RayTracerConfig config)
    : config(std::move(config)) {}

void RefreshRayTracerConfigCommand::execute(Editor* editor)
{
    editor->refreshRayTracerConfig(config);
}