#include "ToggleLiveRayTracingCommand.h"

#include "Editor/Editor.h"

void ToggleLiveRayTracingCommand::execute(Editor* editor)
{
    editor->toggleLiveRayTracing();
}