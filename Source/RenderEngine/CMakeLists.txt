target_sources(Vera-Lib PRIVATE
        Renderer.cpp
        Camera.cpp)

add_subdirectory(Images)
add_subdirectory(RayTracing)
add_subdirectory(SceneObjects)
add_subdirectory(Skybox)
add_subdirectory(Shadows)
