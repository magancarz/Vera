#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"

#include "Objects/Components/MeshComponent.h"

#include "UnitTests/Mocks/MockModel.h"

using testing::Return;

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

TEST_F(MeshComponentTests, shouldFindMaterial)
{
    // given
    MeshComponent mesh_component{&owner};

    auto first_mock_material = std::make_shared<Material>(MaterialInfo{}, "dummy1", nullptr);
    auto second_mock_material = std::make_shared<Material>(MaterialInfo{}, "dummy2", nullptr);
    std::vector<std::shared_ptr<Material>> materials = {first_mock_material, second_mock_material};
    mesh_component.setMaterials(materials);

    // when
    std::shared_ptr<Material> found_material = mesh_component.findMaterial(second_mock_material->getName());

    // then
    EXPECT_EQ(found_material.get(), second_mock_material.get());
}

TEST_F(MeshComponentTests, shouldReturnMeshRequiredMaterials)
{
    // given
    MeshComponent mesh_component{&owner};

    std::shared_ptr<MockModel> mock_model = std::make_shared<MockModel>("dummy");
    std::vector<std::string> expected_required_materials{"dummy_material", "dummy_metal"};
    mesh_component.setModel(mock_model);

    EXPECT_CALL(*mock_model, getRequiredMaterials()).WillOnce(Return(expected_required_materials));

    // when
    const std::vector<std::string> actual_required_materials = mesh_component.getRequiredMaterials();

    // then
    for (size_t i = 0; i < expected_required_materials.size(); ++i)
    {
        EXPECT_EQ(actual_required_materials[i], expected_required_materials[i]);
    }
}