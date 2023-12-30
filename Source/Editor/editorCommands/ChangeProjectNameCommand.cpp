#include "ChangeProjectNameCommand.h"

#include "Editor/Editor.h"

ChangeProjectNameCommand::ChangeProjectNameCommand(std::string project_name)
	: project_name(std::move(project_name)) {}

void ChangeProjectNameCommand::execute(Editor* editor)
{
    editor->changeCurrentProjectName(project_name);
}