#include "gtest/gtest.h"

#include <glm/ext/matrix_clip_space.hpp>
#include "Objects/Components/MeshComponent.h"

#include "UnitTests/Mocks/MockModel.h"

struct MeshComponentTests : public ::testing::Test
{
    Object owner;

    void SetUp() override
    {
        owner = Object{};
    }

    void TearDown() override {}
};

TEST_F(MeshComponentTests, shouldSetModelToComponent)
{
    // given
    MeshComponent mesh_component{&owner};

    auto mock_model = std::make_shared<MockModel>("dummy");

    // when
    mesh_component.setModel(mock_model);

    // then
    EXPECT_EQ(mesh_component.getModel()->getName(), mock_model->getName());
}

TEST_F(MeshComponentTests, shouldSetMaterialsToComponent)
{
    // given
    MeshComponent mesh_component{&owner};

    auto first_mock_material = std::make_shared<Material>(MaterialInfo{}, "dummy1", nullptr);
    auto second_mock_material = std::make_shared<Material>(MaterialInfo{}, "dummy2", nullptr);
    std::vector<std::shared_ptr<Material>> materials = {first_mock_material, second_mock_material};

    // when
    mesh_component.setMaterials(materials);

    // then
    std::vector<std::shared_ptr<Material>> result_materials = mesh_component.getMaterials();
    EXPECT_EQ(result_materials[0]->getName(), first_mock_material->getName());
    EXPECT_EQ(result_materials[1]->getName(), second_mock_material->getName());
}
