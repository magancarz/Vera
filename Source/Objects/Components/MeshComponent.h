#pragma once

#include "Objects/Object.h"
#include "World/World.h"

class MeshComponent : public ObjectComponent
{
public:
    MeshComponent(Object* owner);

    void setModel(std::shared_ptr<Model> in_model);
    std::shared_ptr<Model> getModel() { return model; }

    void setMaterial(std::shared_ptr<Material> in_material);
    std::shared_ptr<Material> getMaterial() { return material; }
    bool isLight() { return material->isLightMaterial(); };

    void createBlasInstance();
    BlasInstance* getBlasInstance() { return &blas_instance; }

private:
    std::shared_ptr<Model> model;
    std::shared_ptr<Material> material;

    BlasInstance blas_instance{};
};
