#include "CreateObjectCommand.h"

#include "Editor/Editor.h"

CreateObjectCommand::CreateObjectCommand(std::shared_ptr<RawModel> model)
    : model(std::move(model)) {}

void CreateObjectCommand::execute(Editor* editor)
{
    editor->createSceneObject(model);
}