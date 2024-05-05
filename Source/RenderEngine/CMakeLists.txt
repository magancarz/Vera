target_sources(Vera-Lib PRIVATE
        Renderer.cpp
        Camera.cpp
        Window.cpp)

add_subdirectory(Materials)
add_subdirectory(Models)
add_subdirectory(PostProcessing)
add_subdirectory(RenderingAPI)
add_subdirectory(SceneRenderers)