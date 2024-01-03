target_sources(Vera-Lib PRIVATE
        Renderer.cpp
        EntityRenderer.cpp
        Camera.cpp)

add_subdirectory(Shadows)
add_subdirectory(Skybox)
add_subdirectory(RayTracing)