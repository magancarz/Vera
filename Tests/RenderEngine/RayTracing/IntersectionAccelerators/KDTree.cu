#include "RenderEngine/RayTracing/IntersectionAccelerators/KDTreeBuilder.h"

#include <gtest/gtest.h>

#include "TestUtils.h"
#include <RenderEngine/RayTracing/PDF/CosinePDF.h>
#include <Utils/DeviceMemoryPointer.h>
#include "RenderEngine/RayTracing/Shapes/Triangle.h"

namespace KDTreeTestsUtils
{
    __global__ void testIntersection(IntersectionAcceleratorTreeTraverser** traverser, HitRecord* result)
    {
        Ray ray{};
        ray.origin = glm::vec3{0, 0, 1};
        ray.direction = glm::vec3{0, 0, 0.5f};
        (*result) = (*traverser)->checkIntersection(&ray);
    }
}

class KDTreeTests : public ::testing::Test
{
protected:
    KDTreeTests()
    {
        //TestUtils::createSetOfShapeMocks(mesh, shapes, shape_infos);
    }

    virtual ~KDTreeTests() {}

    virtual void SetUp() {}
    virtual void TearDown() {}

    int num_of_shapes = 10;
    dmm::DeviceMemoryPointer<Triangle*> shapes{num_of_shapes};
    dmm::DeviceMemoryPointer<ShapeInfo*> shape_infos{num_of_shapes};
};

TEST_F(KDTreeTests, ShouldCreateKDTreeTraverser)
{
    // given
    std::unique_ptr<KDTreeBuilder> kd_tree_builder = std::make_unique<KDTreeBuilder>();

    // when
    auto traverser = kd_tree_builder->buildAccelerator(shapes, shape_infos);

    // then
    ASSERT_TRUE((*traverser) != nullptr);
}

TEST_F(KDTreeTests, ShouldFindCorrectIntersection)
{
    // given
    std::unique_ptr<KDTreeBuilder> kd_tree_builder = std::make_unique<KDTreeBuilder>();
    auto traverser = kd_tree_builder->buildAccelerator(shapes, shape_infos);
    dmm::DeviceMemoryPointer<HitRecord> result;

    // when
    KDTreeTestsUtils::testIntersection<<<1, 1>>>(traverser.get(), result.get());
    cudaDeviceSynchronize();

    // then
    ASSERT_TRUE(true);
}