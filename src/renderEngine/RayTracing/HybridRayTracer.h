#pragma once

#include "RayTracer.h"
#include "RayTracerCamera.h"

class HybridRayTracer : public RayTracer
{
public:
    void runRayTracer(Scene* scene, const std::shared_ptr<RayTracedImage>& current_image, const dim3& blocks, const dim3& threads_per_block) override;
};
