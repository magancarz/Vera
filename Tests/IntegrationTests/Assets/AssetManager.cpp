#include "gtest/gtest.h"

#include <Environment.h>
#include "Assets/AssetManager.h"
#include "Assets/Defines.h"
#include "Mocks/DummyAssetManager.h"
#include "Assets/Model/ModelInfo.h"

TEST(AssetManagerTests, shouldFetchMesh)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};
    const std::string MESH_PATH = Assets::DEFAULT_MODEL_NAME;

    // when
    Mesh* mesh = asset_manager.fetchMesh(MESH_PATH);

    // then
    EXPECT_EQ(mesh->name, "cube");

    EXPECT_EQ(mesh->models.size(), 1);
    EXPECT_EQ(mesh->models[0]->getName(), "cube");

    EXPECT_EQ(mesh->materials.size(), 1);
    EXPECT_EQ(mesh->materials[0]->getName(), "white");
}

TEST(AssetManagerTests, shouldStoreMesh)
{
    // given
    DummyAssetManager dummy_asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};
    std::string mesh_name{"dummy_mesh"};
    auto mesh = std::make_unique<Mesh>(mesh_name);
    Mesh* mesh_address = mesh.get();

    // when
    dummy_asset_manager.storeMesh(std::move(mesh));

    // then
    Mesh* stored_mesh = dummy_asset_manager.getMeshIfAvailable(mesh_name);
    EXPECT_EQ(stored_mesh, mesh_address);
}

TEST(AssetManagerTests, shouldFetchModel)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};
    const std::string MODEL_PATH = Assets::DEFAULT_MODEL_NAME;
    const std::string EXPECTED_MODEL_NAME{"cube"};
    const std::string EXPECTED_REQUIRED_MATERIAL{"white"};

    // when
    const Model* model = asset_manager.fetchModel(MODEL_PATH);

    // then
    EXPECT_EQ(model->getName(), EXPECTED_MODEL_NAME);
    EXPECT_EQ(model->getRequiredMaterial(), EXPECTED_REQUIRED_MATERIAL);
}

TEST(AssetManagerTests, shouldStoreModel)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};
    ModelInfo model_info{};
    model_info.name = "dummy_model";
    model_info.vertices = {Vertex{{2, 0, 0}}, Vertex{{1, 0, 0}}, Vertex{{37, 0, 0}}};
    model_info.indices = {0, 1, 2};
    auto model = std::make_unique<Model>(Environment::memoryAllocator(), model_info);
    Model* model_address = model.get();

    // when
    Model* stored_model = asset_manager.storeModel(std::move(model));

    // then
    EXPECT_EQ(stored_model, model_address);
}

TEST(AssetManagerTests, shouldFetchMaterial)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};

    // when
    const Material* material = asset_manager.fetchMaterial(Assets::DEFAULT_MATERIAL_NAME);

    // then
    EXPECT_EQ(material->getName(), Assets::DEFAULT_MATERIAL_NAME);
    EXPECT_EQ(material->getDiffuseTexture()->getName(), Assets::DEFAULT_DIFFUSE_TEXTURE);
    EXPECT_EQ(material->getNormalTexture()->getName(), Assets::DEFAULT_NORMAL_MAP);
}

TEST(AssetManagerTests, shouldStoreMaterial)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};
    auto material = std::make_unique<Material>(MaterialInfo{.name = "dummy"});
    Material* material_address = material.get();

    // when
    Material* stored_material = asset_manager.storeMaterial(std::move(material));

    // then
    EXPECT_EQ(stored_material, material_address);
}

TEST(AssetManagerTests, shouldFetchDiffuseTexture)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};

    // when
    const DeviceTexture* texture = asset_manager.fetchDiffuseTexture(Assets::DEFAULT_DIFFUSE_TEXTURE);

    // then
    EXPECT_EQ(texture->getName(), Assets::DEFAULT_DIFFUSE_TEXTURE);
}

TEST(AssetManagerTests, shouldFetchNormalMap)
{
    // given
    AssetManager asset_manager{Environment::vulkanHandler(), Environment::memoryAllocator()};

    // when
    const DeviceTexture* texture = asset_manager.fetchNormalMap(Assets::DEFAULT_NORMAL_MAP);

    // then
    EXPECT_EQ(texture->getName(), Assets::DEFAULT_NORMAL_MAP);
}
