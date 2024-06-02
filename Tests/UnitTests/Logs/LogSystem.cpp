#include "gtest/gtest.h"

#include "UnitTests/Mocks/MockLogger.h"
#include "Logs/LogSystem.h"

struct LogSystemTests : public ::testing::Test
{
    void SetUp() override {}
    void TearDown() override
    {
        LogSystem::initialize(nullptr);
    }
};

TEST_F(LogSystemTests, shouldInitializeApplicationLogger)
{
    // given
    auto mock_logger = std::make_unique<MockLogger>();

    // when & then
    LogSystem::initialize(std::move(mock_logger));
}

TEST_F(LogSystemTests, shouldLogMessage)
{
    // given
    auto mock_logger = std::make_unique<MockLogger>();

    // then
    LogSeverity expected_severity = LogSeverity::LOG;
    std::string first_part = "first";
    std::string second_part = "second";
    std::string expected_log = first_part + " " + second_part;
    EXPECT_CALL(*mock_logger, log(expected_severity, testing::StrEq(expected_log))).Times(1);

    // given
    LogSystem::initialize(std::move(mock_logger));

    // when
    LogSystem::log(expected_severity, first_part, " ", second_part);
}