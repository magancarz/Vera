target_sources(Vera-Lib PRIVATE
        Renderer.cpp
        Camera.cpp)

add_subdirectory(HDR)
add_subdirectory(Images)
add_subdirectory(PostProcessing)
add_subdirectory(RayTracing)
add_subdirectory(RenderingUtils)
add_subdirectory(SceneObjects)
add_subdirectory(Skybox)
add_subdirectory(SSAO)
add_subdirectory(Shadows)

target_include_directories(Vera-Lib PUBLIC GLObjects)