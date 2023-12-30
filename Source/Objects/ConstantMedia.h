#pragma once

#include "TriangleMesh.h"

class ConstantMedia : public TriangleMesh {
public:
    ConstantMedia(Scene* parent_scene, std::shared_ptr<RawModel> model_data, Bounds3f bounds, float density);

protected:
    void createShapesOnDeviceMemory() override;
    bool determineIfShapeIsEmittingLight(size_t i) override;

    Bounds3f bounds;
    float density;
};