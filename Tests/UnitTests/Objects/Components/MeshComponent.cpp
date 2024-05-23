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

TEST_F(MeshComponentTests, shouldSetMaterialToComponent)
{
    // given
    MeshComponent mesh_component{&owner};

    auto mock_material = std::make_shared<Material>(MaterialInfo{.color = {0.5, 0.6, 0.7}}, "dummy");

    // when
    mesh_component.setMaterial(mock_material);

    // then
    EXPECT_EQ(mesh_component.getMaterial()->getName(), mock_material->getName());
}
