target_sources(Vera-Lib PRIVATE
        VulkanHandler.cpp
        Pipeline.cpp
        SwapChain.cpp
        CommandPool.cpp
        Device.cpp
        Instance.cpp
        PhysicalDevice.cpp
        VulkanHelper.cpp
        ShaderModule.cpp
        Surface.cpp
        VulkanUtils.cpp)

add_subdirectory(Descriptors)