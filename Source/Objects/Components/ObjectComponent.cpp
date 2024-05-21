#include "ObjectComponent.h"

#include "World/World.h"

ObjectComponent::ObjectComponent(Object* owner, TickGroup tick_group)
    : owner{owner}, tick_group{tick_group} {}

void ObjectComponent::update(FrameInfo& frame_info) {}

void ObjectComponent::setRelativeLocation(const glm::vec3& position)
{
    relative_position = position;
}

glm::vec3 ObjectComponent::getWorldLocation() const
{
    return owner->getLocation() + relative_position;
}