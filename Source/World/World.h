#pragma once

#include "Objects/Components/PlayerMovementComponent.h"
#include "Assets/AssetManager.h"
#include "Project/Project.h"
#include "Objects/Object.h"

class World
{
public:
    void loadProject(const ProjectInfo& project_info, AssetManager& asset_manager);
    void createViewerObject(InputManager& input_manager);

    void update(FrameInfo& frame_info);

    [[nodiscard]] Object* getViewerObject() const { return viewer_object; }
    [[nodiscard]] std::unordered_map<uint32_t, std::unique_ptr<Object>>& getObjects() { return objects; }

protected:
    Object* storeObject(std::unique_ptr<Object> object);

    std::unordered_map<uint32_t, std::unique_ptr<Object>> objects{};
    Object* viewer_object{nullptr};
};
