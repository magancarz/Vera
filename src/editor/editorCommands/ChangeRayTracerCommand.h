#pragma once

#include <memory>

#include "EditorCommand.h"

class RayTracer;

class ChangeRayTracerCommand : public EditorCommand
{
public:
	ChangeRayTracerCommand(std::shared_ptr<RayTracer> new_ray_tracer);

    void execute(Editor* editor) override;

private:
    std::shared_ptr<RayTracer> new_ray_tracer;
};
