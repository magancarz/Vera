#pragma once

#include <string>

#include "GUIElement.h"
#include "RenderEngine/RayTracing/RayTracerConfig.h"

class RayTracerGUIElement : public GUIElement
{
public:
	virtual std::vector<std::shared_ptr<EditorCommand>> renderGUIElement(const EditorInfo& editor_info);

private:
    void printEstimatedTimeOfArrival(const EditorInfo& editor_info);
    std::pair<double, std::string> adjustTimeValue(double time_value);
    void renderImageQueue(const EditorInfo& editor_info, std::vector<std::shared_ptr<EditorCommand>>& editor_commands);
    void renderCurrentlyGeneratedImageQueueElement(const EditorInfo& editor_info, std::vector<std::shared_ptr<EditorCommand>>& editor_commands, RayTracerConfig& image_config);
    void renderImageQueueElement(RayTracerConfig& image_config);

    float ray_tracing_config_input_size = 6.f;

    std::vector<RayTracerConfig> ray_traced_images_queue;
    int image_width = 320;
    int image_height = 200;
    int number_of_ray_bounces = 1;
    int num_of_iterations = 100;
};
