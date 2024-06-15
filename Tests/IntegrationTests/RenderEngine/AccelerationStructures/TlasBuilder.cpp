#include "RenderEngine/AccelerationStructures/TlasBuilder.h"

#include <RenderEngine/RenderingAPI/VulkanDefines.h>
#include <RenderEngine/RenderingAPI/VulkanHelper.h>

#include "gtest/gtest.h"

#include "Environment.h"
#include "TestUtils.h"
#include "Assets/AssetManager.h"
#include "RenderEngine/AccelerationStructures/Blas.h"

TEST(TlasBuilderTests, shouldBuildValidTlas)
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
    AccelerationStructure tlas = Tlas::buildTopLevelAccelerationStructure(
    TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(), blas_instances);

    // then
    EXPECT_NE(tlas.handle, VK_NULL_HANDLE);
    EXPECT_NE(tlas.buffer, nullptr);

    pvkDestroyAccelerationStructureKHR(TestsEnvironment::vulkanHandler().getDeviceHandle(), tlas.handle, VulkanDefines::NO_CALLBACK);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}