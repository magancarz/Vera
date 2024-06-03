#include "Container.h"

#include <iostream>

#include "Logs/LogSystem.h"

Container::Container(std::string component_name)
    : Component(std::move(component_name)) {}

void Container::update(FrameInfo& frame_info)
{
    for (const auto& component : components)
    {
        component->update(frame_info);
    }
}

void Container::addComponent(std::unique_ptr<Component> component)
{
    LogSystem::log(LogSeverity::LOG, "Adding component ", component->getName().c_str(), " to container ", component_name.c_str());
    components.emplace_back(std::move(component));
}

void Container::removeComponent(const Component& component)
{
    auto found = std::ranges::find_if(components.begin(), components.end(),
            [&] (const std::unique_ptr<Component>& item) { return item->getName() == component.getName(); });
    if (found != components.end())
    {
        LogSystem::log(LogSeverity::LOG, "Component ", component.getName().c_str(), " found! Removing from container ", component_name.c_str());
        components.erase(found);
    }
}