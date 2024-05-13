#pragma once

#include <map>

#include "Input/KeyboardMovementController.h"
#include "RenderEngine/Camera.h"
#include "RenderEngine/FrameInfo.h"
#include "Assets/AssetManager.h"
#include "Project/Project.h"

class World
{
public:
    explicit World(Window& window);

    void loadObjects(const ProjectInfo& project_info, AssetManager& asset_manager);

    void update(FrameInfo& frame_info);

    std::map<int, std::shared_ptr<Object>> objects;

private:
    Window& window;

    Camera camera;
    KeyboardMovementController movement_controller{};
    std::unique_ptr<Object> viewer_object;
};
