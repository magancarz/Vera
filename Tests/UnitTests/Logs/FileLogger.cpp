#include "gtest/gtest.h"

#include <filesystem>
#include <TestUtils.h>

#include <Logs/FileLogger.h>
#include <UnitTests/Mocks/MockLogger.h>

#include "Logs/LogSystem.h"
#include "Utils/PathBuilder.h"

struct FileLoggerTests : testing::Test
{
    std::string log_file_location = PathBuilder().append("__test_log__.txt").build();

    void TearDown() override
    {
        TestUtils::deleteFileIfExists(log_file_location);
    }
};

TEST_F(FileLoggerTests, shouldCreateLogFile)
{
    // when
    FileLogger file_logger{log_file_location};

    // then
    EXPECT_TRUE(std::filesystem::exists(log_file_location));
}

TEST_F(FileLoggerTests, shouldLogMessageWithLogSeverity)
{
    // given
    FileLogger file_logger{log_file_location};
    constexpr LogSeverity log_severity = LogSeverity::LOG;
    const std::string log_message{"dummy log"};

    // when
    file_logger.log(log_severity, log_message.c_str());

    // then
    const std::string file_contents = TestUtils::loadFileToString(log_file_location);
    EXPECT_EQ(file_contents, log_message);
}

TEST_F(FileLoggerTests, shouldPassLogMessageToWrapee)
{
    // given
    auto mock_logger = std::make_unique<MockLogger>();
    MockLogger& mock_logger_ref = *mock_logger;
    FileLogger file_logger{std::move(mock_logger), log_file_location};

    constexpr LogSeverity log_severity = LogSeverity::LOG;
    const std::string log_message{"dummy log"};

    // then
    EXPECT_CALL(mock_logger_ref, log(log_severity, testing::StrEq(log_message))).Times(1);

    // when
    file_logger.log(log_severity, log_message.c_str());
}