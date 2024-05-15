#include "gtest/gtest.h"

#include <filesystem>

#include "TestUtils.h"
#include "Project/Project.h"
#include "Utils/PathBuilder.h"

struct ProjectUtilsTests : public ::testing::Test
{
    void SetUp() override {}

    void TearDown() override
    {
        TestUtils::deleteFileIfExists(PROJECT_FILE_LOCATION);
    }

    std::string PROJECT_NAME{"gtest_proj"};
    std::string PROJECT_FILE_LOCATION{PathBuilder(std::filesystem::temp_directory_path()).append(PROJECT_NAME + ProjectUtils::PROJECT_FILE_EXTENSION).build()};
};

TEST_F(ProjectUtilsTests, shouldGenerateProjectFileOnlyWithProjectName)
{
    // given
    ProjectInfo project_info{};
    project_info.project_name = ProjectUtilsTests::PROJECT_NAME;

    std::string expected_file_content{"pm " + ProjectUtilsTests::PROJECT_NAME + "\n"};

    // when
    ProjectUtils::saveProject(project_info, std::filesystem::temp_directory_path().generic_string());

    // then
    EXPECT_TRUE(TestUtils::fileExists(ProjectUtilsTests::PROJECT_FILE_LOCATION));

    std::string file_contents = TestUtils::loadFileToString(ProjectUtilsTests::PROJECT_FILE_LOCATION);
    EXPECT_EQ(file_contents, expected_file_content);
}

TEST_F(ProjectUtilsTests, shouldGenerateProjectFileWithProvidedInformation)
{
    // given
    ProjectInfo project_info{};
    project_info.project_name = ProjectUtilsTests::PROJECT_NAME;

    ObjectInfo object_info{};
    object_info.object_name = "player";
    object_info.model_name = "stanford_dragon";
    object_info.material_name = "white";
    object_info.position = glm::vec3{2, 1, 37};
    object_info.rotation = glm::vec3{0, 1, 0};
    object_info.scale = 0.707;
    project_info.objects_infos.emplace_back(object_info);

    std::string expected_file_content
    {
            "pm " + ProjectUtilsTests::PROJECT_NAME + "\n"
            + "oi " + object_info.toString() + "\n"
    };

    // when
    ProjectUtils::saveProject(project_info, std::filesystem::temp_directory_path().generic_string());

    // then
    EXPECT_TRUE(TestUtils::fileExists(ProjectUtilsTests::PROJECT_FILE_LOCATION));

    std::string file_contents = TestUtils::loadFileToString(ProjectUtilsTests::PROJECT_FILE_LOCATION);
    EXPECT_EQ(file_contents, expected_file_content);
}