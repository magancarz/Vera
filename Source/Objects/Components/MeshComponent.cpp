#include "MeshComponent.h"

MeshComponent::MeshComponent(Object* owner)
    : ObjectComponent(owner) {}

void MeshComponent::setModel(std::shared_ptr<Model> in_model)
{
    model = std::move(in_model);
}

void MeshComponent::setMaterials(std::vector<std::shared_ptr<Material>> in_materials)
{
    materials = std::move(in_materials);
}

std::shared_ptr<Material> MeshComponent::findMaterial(const std::string& name)
{
    auto found_material = std::find_if(materials.begin(), materials.end(),
           [&] (const std::shared_ptr<Material>& material)
    {
        return material->getName() == name;
    });

    return found_material != materials.end() ? *found_material : nullptr;
}

std::vector<std::string> MeshComponent::getRequiredMaterials() const
{
    std::vector<std::string> required_materials;
    std::transform(materials.begin(), materials.end(), std::back_inserter(required_materials),
            [] (const std::shared_ptr<Material>& material)
            {
                return material->getName();
            });

    return required_materials;
}

MeshDescription MeshComponent::getDescription() const
{
    MeshDescription mesh_description{};
    mesh_description.model_descriptions = model->getModelDescriptions();
    mesh_description.required_materials = getRequiredMaterials();
    return mesh_description;
}