#include "gtest/gtest.h"

#include <Logs/BasicLogger.h>

struct BasicLoggerTests : public ::testing::Test
{
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(BasicLoggerTests, shouldLogMessageWithLogSeverity)
{
    // given
    BasicLogger basic_logger{};

    // when & then
    basic_logger.log(LogSeverity::LOG, "dummy log");
}

TEST_F(BasicLoggerTests, shouldLogMessageWithWarningSeverity)
{
    // given
    BasicLogger basic_logger{};

    // when & then
    basic_logger.log(LogSeverity::WARNING, "dummy warning");
}

TEST_F(BasicLoggerTests, shouldLogMessageWithErrorSeverity)
{
    // given
    BasicLogger basic_logger{};

    // when & then
    basic_logger.log(LogSeverity::ERROR, "dummy error");
}

TEST_F(BasicLoggerTests, shouldLogMessageWithFatalSeverity)
{
    // given
    BasicLogger basic_logger{};

    // when & then
    basic_logger.log(LogSeverity::FATAL, "dummy fatal error");
}