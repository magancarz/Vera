#include "Container.h"

#include <iostream>

Container::Container(std::string component_name)
    : Component(std::move(component_name)) {}

void Container::update(FrameInfo& frame_info)
{
    for (const auto& component : components)
    {
        component->update(frame_info);
    }
}

void Container::addComponent(std::shared_ptr<Component> component)
{
    printf("Adding component %s to container %s\n", component->getName().c_str(), component_name.c_str());
    components.emplace_back(std::move(component));
}

void Container::removeComponent(const std::shared_ptr<Component>& component)
{
    auto found = std::find(components.begin(), components.end(),component);
    if (found != components.end())
    {
        printf("Component %s found! Removing from container %s\n", component->getName().c_str(), component_name.c_str());
        components.erase(found);
    }
}