#include <gtest/gtest.h>

#include "TestUtils.h"
#include "RenderEngine/RayTracing/Ray.h"
#include "Utils/CurandUtils.h"
#include "Utils/DeviceMemoryPointer.h"

namespace RandomGeneratorsTestsUtils
{
    __global__ void ShouldReturnRandomFloatInInterval(curandState* curand_state, float min, float max, bool* result)
    {
        const float random_float = randomFloat(curand_state, min, max);
        (*result) = random_float >= min && random_float < max;
    }

    __global__ void ShouldReturnRandomIntInInterval(curandState* curand_state, int min, int max, bool* result)
    {
        const int random_int = randomInt(curand_state, min, max);
        (*result) = random_int >= min && random_int < max;
    }
}

class RandomGeneratorsTests : public ::testing::Test
{
protected:
    RandomGeneratorsTests()
    {
        TestUtils::createCurandState(curand_state.data());
    }

    virtual ~RandomGeneratorsTests() {}

    virtual void SetUp() {}
    virtual void TearDown() {}

    dmm::DeviceMemoryPointer<curandState> curand_state;
    dmm::DeviceMemoryPointer<bool> result;
    float min = 0.f;
    float max = 10.f;
    int num_of_tries = 1000;
};

TEST_F(RandomGeneratorsTests, ShouldReturnRandomFloatInInterval)
{
    for (int i = 0; i < num_of_tries; ++i)
    {
        RandomGeneratorsTestsUtils::ShouldReturnRandomFloatInInterval<<<1, 1>>>(curand_state.data(), min, max, result.data());
        cudaDeviceSynchronize();
        ASSERT_TRUE(*result);
    }
}

TEST_F(RandomGeneratorsTests, ShouldReturnRandomIntInInterval)
{
    for (int i = 0; i < num_of_tries; ++i)
    {
        RandomGeneratorsTestsUtils::ShouldReturnRandomIntInInterval<<<1, 1>>>(curand_state.data(), 0, 2, result.data());
        cudaDeviceSynchronize();
        ASSERT_TRUE(*result);
    }
}