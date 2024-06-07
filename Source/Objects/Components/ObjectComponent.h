#pragma once

struct FrameInfo;
class World;
class Object;

class ObjectComponent
{
public:
    virtual ~ObjectComponent() = default;

    virtual void update(FrameInfo& frame_info) {}

    [[nodiscard]] Object& getOwner() const { return owner; }

protected:
    explicit ObjectComponent(Object& object);

    Object& owner;
};
