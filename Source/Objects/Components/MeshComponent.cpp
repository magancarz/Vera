#include "MeshComponent.h"

#include "Assets/Model/Model.h"
#include "RenderEngine/Materials/Material.h"
#include "Assets/Mesh.h"

MeshComponent::MeshComponent(Object& owner)
    : ObjectComponent(owner) {}

void MeshComponent::setMesh(Mesh* mesh)
{
    name = mesh->name;
    setModels(mesh->models);
    setMaterials(mesh->materials);
}

void MeshComponent::setModels(std::vector<Model*> in_models)
{
    models = std::move(in_models);
    updateModelDescriptions();
    updateRequiredMaterials();
}

void MeshComponent::updateModelDescriptions()
{
    model_descriptions.clear();
    for (auto model : models)
    {
        model_descriptions.emplace_back(model->getModelDescription());
    }
}

void MeshComponent::updateRequiredMaterials()
{
    required_materials.clear();
    for (auto model : models)
    {
        required_materials.emplace_back(model->getRequiredMaterial());
    }
}

void MeshComponent::setMaterials(std::vector<Material*> in_materials)
{
    materials = std::move(in_materials);
}

Material* MeshComponent::findMaterial(const std::string& name) const
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
    return required_materials;
}

MeshDescription MeshComponent::getDescription() const
{
    MeshDescription mesh_description{};
    mesh_description.model_descriptions = model_descriptions;
    mesh_description.required_materials = required_materials;
    return mesh_description;
}