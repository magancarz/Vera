#pragma once

#include "Component.h"
#include "RenderEngine/FrameInfo.h"

class Container : public Component
{
public:
    explicit Container(std::string component_name);

    void update(FrameInfo& frame_info) override;

    void addComponent(std::shared_ptr<Component> component);
    void removeComponent(const std::shared_ptr<Component>& component);

private:
    std::vector<std::shared_ptr<Component>> components;
};
