#pragma once

#include "Component.h"
#include "RenderEngine/FrameInfo.h"

class Container : public Component
{
public:
    explicit Container(std::string component_name);

    void update(FrameInfo& frame_info) override;

    void addComponent(std::unique_ptr<Component> component);
    void removeComponent(const Component& component);

private:
    std::vector<std::unique_ptr<Component>> components;
};
