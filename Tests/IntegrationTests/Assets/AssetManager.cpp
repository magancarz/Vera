#include "gtest/gtest.h"

#include <Environment.h>
#include "Assets/AssetManager.h"
#include "Assets/Defines.h"
#include "Assets/Model/ModelData.h"

TEST(AssetManagerTests, shouldFetchMesh)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const std::string MESH_PATH = Assets::DEFAULT_MESH_NAME;

    // when
    Mesh* mesh = asset_manager.fetchMesh(MESH_PATH);

    // then
    EXPECT_EQ(mesh->name, "cube");

    EXPECT_EQ(mesh->models.size(), 1);
    EXPECT_EQ(mesh->models[0]->getName(), "cube");

    EXPECT_EQ(mesh->materials.size(), 1);
    EXPECT_EQ(mesh->materials[0]->getName(), "white");
}

TEST(AssetManagerTests, shouldFetchModel)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const std::string MODEL_PATH = Assets::DEFAULT_MESH_NAME;
    const std::string EXPECTED_MODEL_NAME{"cube"};
    const std::string EXPECTED_REQUIRED_MATERIAL{"white"};

    // when
    const Model* model = asset_manager.fetchModel(MODEL_PATH);

    // then
    EXPECT_EQ(model->getName(), EXPECTED_MODEL_NAME);
    EXPECT_EQ(model->getRequiredMaterial(), EXPECTED_REQUIRED_MATERIAL);
}

TEST(AssetManagerTests, shouldFetchMaterial)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};

    // when
    const Material* material = asset_manager.fetchMaterial(Assets::DEFAULT_MATERIAL_NAME);

    // then
    EXPECT_EQ(material->getName(), Assets::DEFAULT_MATERIAL_NAME);
    EXPECT_EQ(material->getDiffuseTexture()->getName(), Assets::DEFAULT_DIFFUSE_TEXTURE);
    EXPECT_EQ(material->getNormalTexture()->getName(), Assets::DEFAULT_NORMAL_MAP);
}

TEST(AssetManagerTests, shouldFetchDiffuseTexture)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};

    // when
    const DeviceTexture* texture = asset_manager.fetchDiffuseTexture(Assets::DEFAULT_DIFFUSE_TEXTURE);

    // then
    EXPECT_EQ(texture->getName(), Assets::DEFAULT_DIFFUSE_TEXTURE);
}

TEST(AssetManagerTests, shouldFetchNormalMap)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};

    // when
    const DeviceTexture* texture = asset_manager.fetchNormalMap(Assets::DEFAULT_NORMAL_MAP);

    // then
    EXPECT_EQ(texture->getName(), Assets::DEFAULT_NORMAL_MAP);
}
