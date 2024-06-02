#include <UnitTests/Mocks/MockLogger.h>

#include "gtest/gtest.h"
#include "Logs/LogSystem.h"

int main(int argc, char *argv[])
{
    LogSystem::initialize(std::make_unique<MockLogger>());

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
