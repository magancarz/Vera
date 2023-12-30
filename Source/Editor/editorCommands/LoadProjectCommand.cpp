#include "LoadProjectCommand.h"

#include "Editor/Editor.h"
#include <Utils/CudaErrorChecker.h>

LoadProjectCommand::LoadProjectCommand(std::string project_name)
	: project_name(std::move(project_name)) {}

void LoadProjectCommand::execute(Editor* editor)
{
    editor->loadProject(project_name);
}