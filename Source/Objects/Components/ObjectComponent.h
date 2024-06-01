#pragma once

#include "Objects/TickGroups.h"

struct FrameInfo;
class World;
class Object;

class ObjectComponent
{
public:
    virtual ~ObjectComponent() = default;

    virtual void update(FrameInfo& frame_info);

    [[nodiscard]] Object* getOwner() const { return owner; }
    [[nodiscard]] TickGroup getTickGroup() const { return tick_group; }

protected:
    explicit ObjectComponent(Object* object, TickGroup tick_group = TickGroup::UPDATE);

    Object* owner;
    TickGroup tick_group{TickGroup::UPDATE};
};
