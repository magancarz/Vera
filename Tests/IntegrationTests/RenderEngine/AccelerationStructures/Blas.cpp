#include "gtest/gtest.h"

#include <Environment.h>
#include <TestUtils.h>
#include <Assets/AssetManager.h>
#include <RenderEngine/AccelerationStructures/Blas.h>
#include <RenderEngine/AccelerationStructures/BlasBuilder.h>
#include <RenderEngine/RenderingAPI/VulkanHelper.h>

TEST(BlasTests, shouldBuildValidBlas)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh(Assets::DEBUG_MESH_NAME);

    // when
    const Blas blas{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), asset_manager, *debug_mesh};

    // then
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}

TEST(BlasTests, shouldCreateBlasInstance)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh(Assets::DEBUG_MESH_NAME);

    const Blas blas{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), asset_manager, *debug_mesh};

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