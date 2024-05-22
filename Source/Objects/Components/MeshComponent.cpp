#include "MeshComponent.h"

MeshComponent::MeshComponent(Object* owner)
    : ObjectComponent(owner) {}

void MeshComponent::setModel(std::shared_ptr<Model> in_model)
{
    model = std::move(in_model);
}

void MeshComponent::setMaterial(std::shared_ptr<Material> in_material)
{
    material = std::move(in_material);
}
