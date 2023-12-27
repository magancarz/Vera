#include "Editor.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

#include "helper_cuda.h"
#include "GUI/Display.h"
#include "Scene/Scene.h"
#include "RenderEngine/Camera.h"
#include "Utils/Timer.h"
#include "Input/Input.h"
#include "Editor/EditorCommands/EditorCommand.h"

Editor::Editor()
{
    scene = std::make_shared<Scene>();
    camera = std::make_shared<Camera>(glm::vec3(0, 10, 7));

    loadProject("example_project");
}

void Editor::run()
{
    while (Display::closeNotRequested())
    {
        glfwPollEvents();
        if (!generating_image && camera->move() && toggle_live_ray_tracing)
        {
            resetCurrentRayTracedImage();
        }

        Display::resetInputValues();

        renderScene();

        EditorInfo editor_info = prepareEditorInfo();
        const auto gui_requests = gui_display.renderGUI(editor_info);
        handleEditorCommands(gui_requests);

        checkIfAnyObjectHasBeenSelected();
        updateOutlinedObjects();

        Display::updateDisplay();

        Display::checkCloseRequests();
    }
}

void Editor::renderScene()
{
    if (toggle_live_ray_tracing)
    {
        scene->buildSceneIntersectionAcceleratorIfNeeded();
        ray_tracer.generateRayTracedImage(scene.get(), camera, current_texture);
        master_renderer.renderRayTracedImage(getCurrentRayTracedTexture());

        return;
    }

    if (generating_image)
    {
        scene->buildSceneIntersectionAcceleratorIfNeeded();
        const utils::Timer timer;
        ray_tracer.generateRayTracedImage(scene.get(), camera, current_texture);
        max_time_during_generating += timer.getTimeInMillis();

        master_renderer.renderRayTracedImage(getCurrentRayTracedTexture());

        ++iterations_completed;
        --num_of_iterations_left;
        avg_time_per_ray_traced_sample = max_time_during_generating / iterations_completed;
        eta = num_of_iterations_left * avg_time_per_ray_traced_sample;
        if (num_of_iterations_left <= 0)
        {
            saveImageFromQueue();
            ray_traced_images_queue->erase(ray_traced_images_queue->begin());
            renderNextImageFromQueue();
        }

        return;
    }

    master_renderer.renderScene(camera, scene->lights, scene->triangle_meshes);
}

EditorInfo Editor::prepareEditorInfo()
{
    EditorInfo editor_info{};
    editor_info.scene_objects = &scene->objects;
    editor_info.outlined_objects = &outlined_objects;
    editor_info.generating_image = generating_image;
    editor_info.num_of_iterations_left = num_of_iterations_left;
    editor_info.avg_time_per_image = avg_time_per_ray_traced_sample;
    editor_info.eta = eta;

    return editor_info;
}

void Editor::checkIfAnyObjectHasBeenSelected()
{
    if (Input::isLeftMouseButtonDown())
    {
        if (!Input::isKeyDown(GLFW_KEY_LEFT_CONTROL))
        {
            for (const auto& object : outlined_objects)
            {
                object->setShouldBeOutlined(false);
            }
            outlined_objects.clear();
        }
        scene->buildSceneIntersectionAcceleratorIfNeeded();
        const auto selected_object = ray_tracer.traceRayFromMouse(scene.get(), camera, Display::getMouseX(), Display::getMouseY());
        if (!selected_object.expired() && !selected_object.lock()->shouldBeOutlined())
        {
            selected_object.lock()->setShouldBeOutlined(true);
            outlined_objects.push_back(selected_object.lock().get());
        }
    }
}

void Editor::updateOutlinedObjects()
{
    if (Input::isKeyDown(GLFW_KEY_DELETE))
    {
        for (const auto& object_to_delete : outlined_objects)
        {
            scene->deleteObject(object_to_delete);
        }
        outlined_objects.clear();
    }
}

void Editor::handleEditorCommands(const std::vector<std::shared_ptr<EditorCommand>>& editor_commands)
{
    for (const auto& editor_command : editor_commands)
    {
        editor_command->execute(this);
    }
}

void Editor::createSceneObject(const std::shared_ptr<RawModel>& model)
{
    scene->createTriangleMesh(model);
}

void Editor::clearOutlinedObjectsArray()
{
    for (const auto& object : outlined_objects)
    {
        object->setShouldBeOutlined(false);
    }
    outlined_objects.clear();
}

void Editor::setObjectToBeOutlined(const std::weak_ptr<Object>& object)
{
    outlined_objects.push_back(object.lock().get());
    object.lock()->setShouldBeOutlined(true);
}

void Editor::refreshRayTracerConfig(const RayTracerConfig& config)
{
    refreshCurrentRayTracedImage(config);
}

void Editor::toggleLiveRayTracing()
{
    if (!toggle_live_ray_tracing)
    {
        resetCurrentRayTracedImage();
    }
    toggle_live_ray_tracing = !toggle_live_ray_tracing;
}

void Editor::saveRayTracedImageToFile(const std::string& output_path)
{
    saveImageToFile(output_path);
}

void Editor::generateRayTracedImagesQueue(std::vector<RayTracerConfig>* queue)
{
    ray_traced_images_queue = queue;
    renderNextImageFromQueue();
}

void Editor::renderNextImageFromQueue()
{
    if (!ray_traced_images_queue->empty())
    {
        generating_image = true;
        refreshCurrentRayTracedImage(ray_traced_images_queue->at(0));
        num_of_iterations_left = ray_traced_images_queue->at(0).number_of_iterations;
        iterations_completed = 0;
    	max_time_during_generating = 0.0;
        avg_time_per_ray_traced_sample = 0.0;
        eta = 0;
        return;
    }
    generating_image = false;
}

void Editor::saveImageFromQueue()
{
    saveRayTracedImageToFile("Results/" + ray_traced_images_queue->at(0).image_name);
}

void Editor::stopGeneratingRayTracedImage()
{
    generating_image = false;
}

void Editor::saveCurrentProject()
{
    updateCurrentProjectInfo();
    ProjectUtils::saveProject(current_project_info);
}

void Editor::updateCurrentProjectInfo()
{
    current_project_info.objects_infos = scene->gatherObjectsInfos();
}

void Editor::loadProject(const std::string& project_name)
{
    current_project_info = ProjectUtils::loadProject(project_name);
    scene->loadSceneFromProject(current_project_info);
}

void Editor::changeCurrentProjectName(const std::string& project_name)
{
    current_project_info.project_name = project_name;
}

void Editor::resetCurrentRayTracedImage()
{
    current_texture->clearImage();
}

void Editor::refreshCurrentRayTracedImage(const RayTracerConfig& config)
{
    current_texture.reset();
    current_texture = std::make_shared<RayTracedImage>(config);
}

void Editor::saveImageToFile(const std::string& path)
{
    current_texture->saveImageToFile(path);
}
