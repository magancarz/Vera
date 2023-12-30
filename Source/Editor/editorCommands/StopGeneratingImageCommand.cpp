#include "StopGeneratingImageCommand.h"

#include "Editor/Editor.h"

void StopGeneratingImageCommand::execute(Editor* editor)
{
    editor->stopGeneratingRayTracedImage();
}