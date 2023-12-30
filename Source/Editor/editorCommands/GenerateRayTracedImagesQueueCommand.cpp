#include "GenerateRayTracedImagesQueueCommand.h"

#include "Editor/Editor.h"

GenerateRayTracedImagesQueueCommand::GenerateRayTracedImagesQueueCommand(std::vector<RayTracerConfig>* ray_traced_images_queue)
    : ray_traced_images_queue(ray_traced_images_queue) {}

void GenerateRayTracedImagesQueueCommand::execute(Editor* editor)
{
    editor->generateRayTracedImagesQueue(ray_traced_images_queue);
}