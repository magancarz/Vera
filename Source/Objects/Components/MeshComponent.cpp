#include "MeshComponent.h"

#include "RenderEngine/Models/Model.h"
#include "RenderEngine/Materials/Material.h"

MeshComponent::MeshComponent(Object& owner)
    : ObjectComponent(owner) {}

void MeshComponent::setModel(Model* in_model)
{
    model = in_model;
}

void MeshComponent::setMaterials(std::vector<Material*> in_materials)
{
    materials = std::move(in_materials);
}

Material* MeshComponent::findMaterial(const std::string& name)
{
    const auto found_material = std::ranges::find_if(materials.begin(), materials.end(),
           [&] (const Material* material)
    {
        return material->getName() == name;
    });

    return found_material != materials.end() ? *found_material : nullptr;
}

std::vector<std::string> MeshComponent::getRequiredMaterials() const
{
    return model->getRequiredMaterials();
}

MeshDescription MeshComponent::getDescription() const
{
    MeshDescription mesh_description{};
    mesh_description.model_descriptions = model->getModelDescriptions();
    mesh_description.required_materials = getRequiredMaterials();
    return mesh_description;
}