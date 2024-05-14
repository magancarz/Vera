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

TEST_F(ProjectUtilsTests, shouldGenerateEmptyProjectFile)
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