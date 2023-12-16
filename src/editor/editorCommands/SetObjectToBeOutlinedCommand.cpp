#include "SetObjectToBeOutlinedCommand.h"

#include "Editor/Editor.h"

SetObjectToBeOutlinedCommand::SetObjectToBeOutlinedCommand(std::weak_ptr<Object> object_to_be_outlined)
    : object_to_be_outlined(object_to_be_outlined) {}

void SetObjectToBeOutlinedCommand::execute(Editor* editor)
{
    editor->setObjectToBeOutlined(object_to_be_outlined);
}