#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../GUI/GUI.h"
#include "../RenderEngine/Renderer.h"
#include "Project/Project.h"
#include "RenderEngine/RayTracing/RayTracer.h"

class Scene;
struct RayTracerConfig;
class SceneObjectFactory;
struct RayTracedImageInfo;
class Camera;
class Object;
class IntersectionAcceleratorTreeBuilder;

struct EditorInfo
{
    std::vector<std::shared_ptr<Object>>* scene_objects;
    std::vector<Object*>* outlined_objects;
    bool generating_image = false;
    int num_of_iterations_left = 0;
    double avg_time_per_image = 0.0;
    double eta = 0.0;
};

class Editor
{
public:
    Editor();

    void run();

    void createSceneObject(const std::shared_ptr<RawModel>& model);
    void clearOutlinedObjectsArray();
    void setObjectToBeOutlined(const std::weak_ptr<Object>& object);
    void refreshRayTracerConfig(const RayTracerConfig& config);
    void toggleLiveRayTracing();
    void saveRayTracedImageToFile(const std::string& output_path);
    void generateRayTracedImagesQueue(std::vector<RayTracerConfig>* queue);
    void renderNextImageFromQueue();
    void saveImageFromQueue();
    void stopGeneratingRayTracedImage();
    void saveCurrentProject();
    void updateCurrentProjectInfo();
    void loadProject(const std::string& project_name);
    void changeCurrentProjectName(const std::string& project_name);

private:
    void renderScene();
    EditorInfo prepareEditorInfo();
    void checkIfAnyObjectHasBeenSelected();
    void updateOutlinedObjects();
    void handleEditorCommands(const std::vector<std::shared_ptr<EditorCommand>>& editor_commands);
    void resetCurrentRayTracedImage();
    void refreshCurrentRayTracedImage(const RayTracerConfig& config);
    void saveImageToFile(const std::string& path);
    unsigned int getCurrentRayTracedTexture() const { return current_texture->texture.texture_id; }

    GUI gui_display;
    Renderer master_renderer;
    std::shared_ptr<RayTracer> ray_tracer;
    std::shared_ptr<Camera> camera;

    std::shared_ptr<Scene> scene;
    std::vector<Object*> outlined_objects;
    ProjectInfo current_project_info;

    std::vector<RayTracerConfig>* ray_traced_images_queue;
    std::shared_ptr<RayTracedImage> current_texture;

    bool toggle_live_ray_tracing = false;
    bool generating_image = false;
    int iterations_completed = 0;
    int num_of_iterations_left = 0;
    double max_time_during_generating = 0.0;
    double avg_time_per_ray_traced_sample = 0.0;
    double eta = 0;
};
