#pragma once

#include <map>

#include "Input/KeyboardMovementController.h"
#include "RenderEngine/Camera.h"
#include "RenderEngine/FrameInfo.h"

class World
{
public:
    World(Window& window);

    void loadObjects(Device& device/*, const std::vector<std::shared_ptr<Material>>& available_materials*/);

    void update(FrameInfo& frame_info);

private:
    Window& window;

    Camera camera;
    KeyboardMovementController movement_controller{};
    std::unique_ptr<Object> viewer_object;

    std::map<int, std::shared_ptr<Object>> objects;
};
