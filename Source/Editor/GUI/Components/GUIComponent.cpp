#include "GUIComponent.h"

#include "Logs/LogSystem.h"

GUIComponent::GUIComponent(std::string name)
    : component_name{std::move(name)}
{
    LogSystem::log(LogSeverity::LOG, "Creating component ", component_name.c_str());
}
