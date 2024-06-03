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
    void loadProject(const ProjectInfo& project_info, AssetManager& asset_manager);
    void createViewerObject(InputManager& input_manager);

    void update(FrameInfo& frame_info);

    void registerComponent(std::weak_ptr<ObjectComponent> component);

    [[nodiscard]] std::unique_ptr<Object>& getViewerObject() { return viewer_object; }
    [[nodiscard]] std::unordered_map<int, std::unique_ptr<Object>>& getRenderedObjects() { return rendered_objects; }

protected:
    std::unique_ptr<Object> viewer_object{nullptr};
    std::unordered_map<int, std::unique_ptr<Object>> rendered_objects{};

    void removeUnusedRegisteredComponents();
    void updateRegisteredComponents(FrameInfo& frame_info);

    std::map<TickGroup, std::vector<std::weak_ptr<ObjectComponent>>> registered_components{};
};
