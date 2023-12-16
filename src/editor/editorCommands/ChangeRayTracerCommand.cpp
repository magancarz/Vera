#include "ChangeRayTracerCommand.h"

#include "Editor/Editor.h"

ChangeRayTracerCommand::ChangeRayTracerCommand(std::shared_ptr<RayTracer> new_ray_tracer)
    : new_ray_tracer(std::move(new_ray_tracer)) {}

void ChangeRayTracerCommand::execute(Editor* editor)
{
    editor->changeRayTracer(new_ray_tracer);
}