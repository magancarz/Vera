target_sources(Vera-Lib PRIVATE
        Renderer.cpp)

add_subdirectory(AccelerationStructures)
add_subdirectory(PostProcessing)
add_subdirectory(RenderingAPI)
add_subdirectory(SceneRenderers)
add_subdirectory(Textures)