#pragma once

#include <memory>

#include "GUIComponent.h"
#include "RenderEngine/FrameInfo.h"

class GUIContainer : public GUIComponent
{
public:
    explicit GUIContainer(std::string component_name);

    void update(FrameInfo& frame_info) override;

    void addComponent(std::unique_ptr<GUIComponent> component);
    void removeComponent(const GUIComponent& component);

private:
    std::vector<std::unique_ptr<GUIComponent>> components;
};
