#include "CreateLightCommand.h"

#include "Editor/Editor.h"

CreateLightCommand::CreateLightCommand(std::shared_ptr<LightCreator> light_creator)
    : light_creator(std::move(light_creator)) {}

void CreateLightCommand::execute(Editor* editor)
{
    editor->createSceneLight(light_creator);
}