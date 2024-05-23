#pragma once

#include "Objects/Object.h"
#include "World/World.h"
#include "RenderEngine/RenderingAPI/Blas.h"

class MeshComponent : public ObjectComponent
{
public:
    explicit MeshComponent(Object* owner);

    void setModel(std::shared_ptr<Model> in_model);
    std::shared_ptr<Model> getModel() { return model; }

    void setMaterial(std::shared_ptr<Material> in_material);
    std::shared_ptr<Material> getMaterial() { return material; }
    bool isLight() { return material->isLightMaterial(); };

private:
    std::shared_ptr<Model> model;
    std::shared_ptr<Material> material;
};
