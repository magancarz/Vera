#include "ClearOutlinedObjectsCommand.h"

#include "Editor/Editor.h"

void ClearOutlinedObjectsCommand::execute(Editor* editor)
{
    editor->clearOutlinedObjectsArray();
}