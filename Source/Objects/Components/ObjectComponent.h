#pragma once

#include "RenderEngine/FrameInfo.h"
#include "Objects/TickGroups.h"

class World;
class Object;

class ObjectComponent
{
public:
    virtual ~ObjectComponent() = default;

    virtual void update(FrameInfo& frame_info);

    void setRelativeLocation(const glm::vec3& position);
    [[nodiscard]] glm::vec3 getRelativeLocation() const { return relative_position; }
    [[nodiscard]] glm::vec3 getWorldLocation() const;

    Object* getOwner() { return owner; }
    [[nodiscard]] TickGroup getTickGroup() const { return tick_group; }

protected:
    explicit ObjectComponent(Object* object, TickGroup tick_group = TickGroup::UPDATE);

    Object* owner;
    TickGroup tick_group{TickGroup::UPDATE};

    glm::vec3 relative_position{0};
};
