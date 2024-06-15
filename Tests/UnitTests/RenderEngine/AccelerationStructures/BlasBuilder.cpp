#include "gtest/gtest.h"

#include <Environment.h>
#include <TestUtils.h>
#include <Assets/AssetManager.h>

#include <vulkan/vulkan.hpp>

#include <Assets/Model/Vertex.h>
#include <RenderEngine/AccelerationStructures/AccelerationStructure.h>
#include <RenderEngine/AccelerationStructures/BlasBuilder.h>
#include <RenderEngine/RenderingAPI/VulkanDefines.h>
#include <RenderEngine/RenderingAPI/VulkanHelper.h>

TEST(BlasBuilderTests, shouldBuildValidBlas)
{
    // given
    AssetManager asset_manager{TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator()};
    const Model* debug_model = asset_manager.fetchModel(Assets::DEBUG_MESH_NAME);
    ModelDescription model_description = debug_model->getModelDescription();

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = model_description.vertex_buffer->getBufferDeviceAddress();
    triangles.vertexStride = sizeof(Vertex);

    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = model_description.index_buffer->getBufferDeviceAddress();

    triangles.maxVertex = static_cast<uint32_t>(model_description.num_of_triangles * 3 - 1);

    VkAccelerationStructureGeometryKHR acceleration_structure_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

    acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    acceleration_structure_geometry.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex = 0;
    offset.primitiveCount = static_cast<uint32_t>(model_description.num_of_triangles);
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    BlasBuilder::BlasInput blas_input{};
    blas_input.acceleration_structure_geometry.emplace_back(acceleration_structure_geometry);
    blas_input.acceleration_structure_build_offset_info.emplace_back(offset);
    blas_input.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

    std::vector<BlasBuilder::BlasInput> blas_inputs{blas_input};

    // when
    std::vector<AccelerationStructure> blases = BlasBuilder::buildBottomLevelAccelerationStructures(
        TestsEnvironment::vulkanHandler(), TestsEnvironment::memoryAllocator(),
        blas_inputs, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // then
    EXPECT_EQ(blases.size(), blas_inputs.size());

    AccelerationStructure blas = std::move(blases.front());
    EXPECT_NE(blas.handle, VK_NULL_HANDLE);
    EXPECT_NE(blas.buffer, nullptr);

    pvkDestroyAccelerationStructureKHR(TestsEnvironment::vulkanHandler().getDeviceHandle(), blas.handle, VulkanDefines::NO_CALLBACK);
    TestUtils::failIfVulkanValidationLayersErrorsWerePresent();
}