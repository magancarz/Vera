#include "gtest/gtest.h"

#include <Environment.h>
#include <TestUtils.h>
#include <Assets/AssetManager.h>
#include <RenderEngine/AccelerationStructures/Blas/Blas.h>
#include <RenderEngine/AccelerationStructures/Blas/DynamicBlas.h>
#include <RenderEngine/RenderingAPI/VulkanHelper.h>

TEST(BlasTests, shouldBuildValidBlas)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh(Assets::DEBUG_MESH_NAME);

    DynamicBlas blas{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), asset_manager, *debug_mesh};

    // when
    blas.createBlas();

    // then
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}

TEST(BlasTests, shouldCreateBlasInstance)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh(Assets::DEBUG_MESH_NAME);

    DynamicBlas blas{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), asset_manager, *debug_mesh};
    blas.createBlas();

    const glm::mat4 transform = TestUtils::randomTransform();

    // when
    const BlasInstance blas_instance = blas.createBlasInstance(transform);

    // then
    TestUtils::expectTwoMatricesToBeEqual(
        blas_instance.bottom_level_acceleration_structure_instance.transform, VulkanHelper::mat4ToVkTransformMatrixKHR(transform));
    EXPECT_TRUE(blas_instance.bottom_level_geometry_instance_buffer != nullptr);
    EXPECT_NE(blas_instance.bottom_level_geometry_instance_device_address, 0ULL);

    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}