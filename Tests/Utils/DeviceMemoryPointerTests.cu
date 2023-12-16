#include <gtest/gtest.h>

#include "TestUtils.h"
#include "Utils/DeviceMemoryPointer.h"

TEST(DeviceMemoryPointerTests, ShouldCopyCorrectValueFromOtherPointer)
{
    // given
    constexpr int value = 10;
    dmm::DeviceMemoryPointer<int> integer;

    // when
    integer.copyFrom(&value);

    // then
    EXPECT_TRUE((*integer) == value);
}

TEST(DeviceMemoryPointerTests, ShouldCopyCorrectValueToOtherObjects)
{
    // given
    constexpr int value = 10;
    dmm::DeviceMemoryPointer<int> integer;
    integer.copyFrom(&value);

    // when
    const int actual = *integer;

    // then
    EXPECT_TRUE(actual == value);
}

TEST(DeviceMemoryPointerTests, ShouldFreeMemoryAfterUseCountIs0)
{
    // given
    constexpr int value = 10;
    int* ptr;

    // when
    {
        dmm::DeviceMemoryPointer<int> integer;
        integer.copyFrom(&value);
        auto dmm1 = integer;
        auto dmm2 = std::move(dmm1);
        auto dmm3 = dmm::DeviceMemoryPointer<int>(dmm2);
        auto dmm4 = dmm::DeviceMemoryPointer<int>(std::move(dmm3));
        ptr = dmm4.data();
    }

    // then
    int actual;
    cudaMemcpy(&actual, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(actual != value);
}

TEST(DeviceMemoryPointerTests, ShouldNotFreeMemoryIfUseCountIsGreaterThan0)
{
    // given
    int* ptr;
    constexpr int value = 10;
    dmm::DeviceMemoryPointer<int> dmm1;
    dmm1.copyFrom(&value);
    ptr = dmm1.data();

    // when
    {
        dmm::DeviceMemoryPointer<int> dmm2 = dmm1;
    }

    // then
    EXPECT_TRUE((*ptr) == 10);
}