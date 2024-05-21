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

void MeshComponent::createBlasInstance()
{
    assert(model != nullptr && material != nullptr);

    auto transform_component = owner->findComponentByClass<TransformComponent>();
    blas_instance = model->createBlasInstance(transform_component->transform(), owner->getID());
    material->assignMaterialHitGroup(blas_instance);
}
