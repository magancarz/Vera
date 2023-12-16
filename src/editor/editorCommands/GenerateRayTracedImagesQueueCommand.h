#pragma once

#include <memory>
#include <vector>

#include "EditorCommand.h"
#include "RenderEngine/RayTracing/RayTracerConfig.h"

class GenerateRayTracedImagesQueueCommand : public EditorCommand
{
public:
	GenerateRayTracedImagesQueueCommand(std::vector<RayTracerConfig>* ray_traced_images_queue);

    void execute(Editor* editor) override;

private:
    std::vector<RayTracerConfig>* ray_traced_images_queue;
};
