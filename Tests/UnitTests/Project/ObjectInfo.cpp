#include "gtest/gtest.h"

#include "Project/ObjectInfo.h"

struct ObjectInfoTests : public ::testing::Test
{
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ObjectInfoTests, shouldGenerateCorrectStringWithGivenObjectInfo)
{
    // given
    ObjectInfo object_info{};
    object_info.object_name = "player";
    object_info.model_name = "stanford_dragon";
    object_info.material_name = "white";
    object_info.position = glm::vec3{2, 1, 37};
    object_info.rotation = glm::vec3{0, 0.71558499f, 0};
    object_info.scale = 0.707;

    const std::string expected_string{"player stanford_dragon white 2.000000 1.000000 37.000000 0.000000 41.000000 0.000000 0.707000"};

    // when
    const std::string object_info_as_string = object_info.toString();

    // then
    EXPECT_EQ(object_info_as_string, expected_string);
}

TEST_F(ObjectInfoTests, shouldExtractObjectInfoFromGivenString)
{
    // given
    const std::string object_info_as_string{"player stanford_dragon white 2.000000 1.000000 37.000000 0.000000 41.000000 0.000000 0.707000"};

    ObjectInfo expected_object_info{};
    expected_object_info.object_name = "player";
    expected_object_info.model_name = "stanford_dragon";
    expected_object_info.material_name = "white";
    expected_object_info.position = glm::vec3{2, 1, 37};
    expected_object_info.rotation = glm::vec3{0, glm::radians(41.0f), 0};
    expected_object_info.scale = 0.707;

    // when
    ObjectInfo extracted_object_info = ObjectInfo::fromString(object_info_as_string);

    // then
    EXPECT_EQ(extracted_object_info.object_name, expected_object_info.object_name);
    EXPECT_EQ(extracted_object_info.model_name, expected_object_info.model_name);
    EXPECT_EQ(extracted_object_info.material_name, expected_object_info.material_name);
    EXPECT_EQ(extracted_object_info.position, expected_object_info.position);
    EXPECT_EQ(extracted_object_info.rotation, expected_object_info.rotation);
    EXPECT_EQ(extracted_object_info.scale, expected_object_info.scale);
}