#include <gtest/gtest.h>

#include "TestUtils.h"
#include <RenderEngine/RayTracing/PDF/CosinePDF.h>
#include <Utils/DeviceMemoryPointer.h>

class CosinePDFTests : public ::testing::Test
{
protected:
    CosinePDFTests()
    {
        TestUtils::createCurandState(curand_state.data());
    }

    virtual ~CosinePDFTests() {}

    virtual void SetUp() {}
    virtual void TearDown() {}

    dmm::DeviceMemoryPointer<curandState> curand_state;
};

TEST_F(CosinePDFTests, SimpleTest)
{
    //CosinePDF cosine_pdf{curand_state.get(), glm::vec3{0, 1, 0}};
    ASSERT_TRUE(true);
}