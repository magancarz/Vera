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
    explicit MeshComponent(Object* owner);

    void setModel(std::shared_ptr<Model> in_model);
    std::shared_ptr<Model> getModel() { return model; }

    void setMaterials(std::vector<std::shared_ptr<Material>> in_materials);
    std::vector<std::shared_ptr<Material>> getMaterials() { return materials; }
    std::shared_ptr<Material> findMaterial(const std::string& name);
    [[nodiscard]] std::vector<std::string> getRequiredMaterials() const;

    [[nodiscard]] MeshDescription getDescription() const;

private:
    std::shared_ptr<Model> model{nullptr};
    std::vector<std::shared_ptr<Material>> materials{};
};
