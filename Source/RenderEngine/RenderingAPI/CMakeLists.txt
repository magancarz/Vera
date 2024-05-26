target_sources(Vera-Lib PRIVATE
        Blas.cpp
        VulkanFacade.cpp
        Pipeline.cpp
        SwapChain.cpp
        CommandPool.cpp
        Descriptors.cpp
        Device.cpp
        Instance.cpp
        PhysicalDevice.cpp
        VulkanHelper.cpp
        ShaderModule.cpp
        Surface.cpp)

add_subdirectory(Textures)