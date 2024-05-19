#pragma once

#include "RenderEngine/FrameInfo.h"
#include "Objects/TickGroups.h"

class World;
class Object;

class ObjectComponent
{
public:
    explicit ObjectComponent(Object* owner, World* world, TickGroup tick_group = TickGroup::UPDATE);
    virtual ~ObjectComponent() = default;

    virtual void update(FrameInfo& frame_info) = 0;

    void setRelativeLocation(const glm::vec3& position);
    [[nodiscard]] glm::vec3 getRelativeLocation() const { return relative_position; }
    [[nodiscard]] glm::vec3 getWorldLocation() const;

    [[nodiscard]] TickGroup getTickGroup() const { return tick_group; }

protected:
    Object* owner;
    TickGroup tick_group{TickGroup::UPDATE};

    glm::vec3 relative_position{0};
};
