target_sources(Vera-Lib PRIVATE
        Renderer.cpp
        Window.cpp)

add_subdirectory(Materials)
add_subdirectory(Models)
add_subdirectory(PostProcessing)
add_subdirectory(RenderingAPI)
add_subdirectory(SceneRenderers)