#include "SaveProjectCommand.h"

#include "Editor/Editor.h"

void SaveProjectCommand::execute(Editor* editor)
{
    editor->saveCurrentProject();
}