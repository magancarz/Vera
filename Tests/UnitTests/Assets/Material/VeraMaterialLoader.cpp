#include "gtest/gtest.h"

#include <Assets/Defines.h>
#include <Assets/Material/VeraMaterialLoader.h>

TEST(VeraMaterialLoaderTests, shouldCorrectlyLoadMaterialData)
{
    // given
    std::string example_vera_material_name = Assets::DEBUG_MATERIAL_NAME;

    // when
    MaterialData material_data = VeraMaterialLoader::loadAssetFromFile(example_vera_material_name);

    // then
    EXPECT_EQ(material_data.name, Assets::DEBUG_MATERIAL_NAME);
    EXPECT_EQ(material_data.diffuse_texture_name, Assets::DEBUG_DIFFUSE_TEXTURE_NAME);
    EXPECT_EQ(material_data.normal_map_name, Assets::DEBUG_NORMAL_MAP_NAME);
}

TEST(VeraMaterialLoaderTests, shouldReturnDefaultMaterialWhenFileDoesntExist)
{
    // given
    std::string example_non_existing_vera_material_name{"__invalid_texdasdasdasdad__"};

    // when
    MaterialData material_data = VeraMaterialLoader::loadAssetFromFile(example_non_existing_vera_material_name);

    // then
    MaterialData default_material{};
    EXPECT_EQ(material_data.name, default_material.name);
    EXPECT_EQ(material_data.diffuse_texture_name, default_material.diffuse_texture_name);
    EXPECT_EQ(material_data.normal_map_name, default_material.normal_map_name);
}