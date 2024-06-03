#pragma once

#include <vector>
#include <memory>

#include "ObjectComponent.h"
#include "RenderEngine/Models/MeshDescription.h"

class Model;
class Material;

class MeshComponent : public ObjectComponent
{
public:
    explicit MeshComponent(Object& owner);

    void setModel(Model* in_model);
    Model* getModel() const { return model; }

    void setMaterials(std::vector<Material*> in_materials);
    std::vector<Material*> getMaterials() { return materials; }
    Material* findMaterial(const std::string& name);
    [[nodiscard]] std::vector<std::string> getRequiredMaterials() const;

    [[nodiscard]] MeshDescription getDescription() const;

private:
    Model* model{nullptr};
    std::vector<Material*> materials{};
};
