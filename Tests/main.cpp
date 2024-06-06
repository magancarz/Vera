#include "gtest/gtest.h"

#include <Environment.h>

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new Environment());
    return RUN_ALL_TESTS();
}
