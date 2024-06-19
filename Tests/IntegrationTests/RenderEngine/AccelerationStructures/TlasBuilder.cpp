#include "RenderEngine/AccelerationStructures/Tlas.h"

#include <RenderEngine/RenderingAPI/VulkanHelper.h>

#include "gtest/gtest.h"

#include "Environment.h"
#include "TestUtils.h"
#include "Assets/AssetManager.h"
#include "RenderEngine/AccelerationStructures/Blas.h"

TEST(TlasTests, shouldBuildValidTlas)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Mesh* debug_mesh = asset_manager.fetchMesh(Assets::DEBUG_MESH_NAME);

    const Blas blas{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), asset_manager, *debug_mesh};
    const glm::mat4 transform = TestUtils::randomTransform();
    BlasInstance blas_instance = blas.createBlasInstance(transform);
    std::vector<BlasInstance> blas_instances{};
    blas_instances.emplace_back(std::move(blas_instance));

    // when
    Tlas tlas{
        TestsEnvironment::vulkanHandler().getLogicalDevice(),
        TestsEnvironment::vulkanHandler().getCommandPool(),
        TestsEnvironment::memoryAllocator(),
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        blas_instances};

    // then
    const AccelerationStructure& acceleration_structure = tlas.accelerationStructure();
    EXPECT_NE(acceleration_structure.getHandle(), VK_NULL_HANDLE);

    const Buffer& buffer = acceleration_structure.getBuffer();
    EXPECT_NE(buffer.getBuffer(), VK_NULL_HANDLE);

    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}