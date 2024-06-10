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
    Object* addObject(std::unique_ptr<Object> object);

    template <typename T>
    T* addObject(std::unique_ptr<T> object)
    {
        static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object class");
        T* out_ptr = object.get();
        addObject(std::unique_ptr<Object>(std::move(object)));
        return out_ptr;
    }

    void update(FrameInfo& frame_info);

    [[nodiscard]] Object* getViewerObject() const { return viewer_object; }
    [[nodiscard]] std::unordered_map<uint32_t, std::unique_ptr<Object>>& getObjects() { return objects; }

protected:

    std::unordered_map<uint32_t, std::unique_ptr<Object>> objects{};
    Object* viewer_object{nullptr};
};
