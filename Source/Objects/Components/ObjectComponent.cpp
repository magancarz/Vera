#include "ObjectComponent.h"

ObjectComponent::ObjectComponent(Object& owner, TickGroup tick_group)
    : owner{owner}, tick_group{tick_group} {}
