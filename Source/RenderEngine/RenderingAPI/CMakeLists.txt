target_sources(Vera-Lib PRIVATE
        VulkanFacade.cpp
        Pipeline.cpp
        SwapChain.cpp
        Buffer.cpp
        Descriptors.cpp
        Device.cpp
        Instance.cpp
        VulkanHelper.cpp
        ShaderModule.cpp)

add_subdirectory(Textures)