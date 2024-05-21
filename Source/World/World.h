#pragma once

#include <map>

#include "Objects/Components/PlayerMovementComponent.h"
#include "RenderEngine/FrameInfo.h"
#include "Assets/AssetManager.h"
#include "Project/Project.h"

class Object;

class World
{
public:
    World() = default;

    void loadProject(const ProjectInfo& project_info, const std::shared_ptr<AssetManager>& asset_manager);

    void update(FrameInfo& frame_info);

    void registerComponent(std::weak_ptr<ObjectComponent> component);

    std::map<int, std::shared_ptr<Object>> rendered_objects;

protected:
    void loadViewerObject();

    std::shared_ptr<Object> viewer_object;

    void removeUnusedRegisteredComponents();
    void updateRegisteredComponents(FrameInfo& frame_info);

    std::map<TickGroup, std::vector<std::weak_ptr<ObjectComponent>>> registered_components;
};
