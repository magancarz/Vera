target_sources(Vera-Lib PRIVATE
        VulkanFacade.cpp
        Pipeline.cpp
        SwapChain.cpp
        Buffer.cpp
        CommandPool.cpp
        Descriptors.cpp
        Device.cpp
        Instance.cpp
        PhysicalDevice.cpp
        VulkanHelper.cpp
        ShaderModule.cpp
        Surface.cpp)

add_subdirectory(Textures)