#include "GUIContainer.h"

#include <iostream>

#include "Logs/LogSystem.h"

GUIContainer::GUIContainer(std::string component_name)
    : GUIComponent(std::move(component_name)) {}

void GUIContainer::update(FrameInfo& frame_info)
{
    for (const auto& component : components)
    {
        component->update(frame_info);
    }
}

void GUIContainer::addComponent(std::unique_ptr<GUIComponent> component)
{
    LogSystem::log(LogSeverity::LOG, "Adding component ", component->getName().c_str(), " to container ", component_name.c_str());
    components.emplace_back(std::move(component));
}

void GUIContainer::removeComponent(const GUIComponent& component)
{
    auto found = std::ranges::find_if(components.begin(), components.end(),
            [&] (const std::unique_ptr<GUIComponent>& item) { return item->getName() == component.getName(); });
    if (found != components.end())
    {
        LogSystem::log(LogSeverity::LOG, "Component ", component.getName().c_str(), " found! Removing from container ", component_name.c_str());
        components.erase(found);
    }
}