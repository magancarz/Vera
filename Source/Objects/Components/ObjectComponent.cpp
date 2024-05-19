#include "ObjectComponent.h"

#include "World/World.h"

ObjectComponent::ObjectComponent(Object* owner, World* world, TickGroup tick_group)
    : owner{owner}, tick_group{tick_group}
{
    world->registerComponent(this);
}

void ObjectComponent::setRelativeLocation(const glm::vec3& position)
{
    relative_position = position;
}

glm::vec3 ObjectComponent::getWorldLocation() const
{
    return owner->getLocation() + relative_position;
}