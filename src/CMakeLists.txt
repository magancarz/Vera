target_sources(Tiny-Ray-Tracing-Lib PRIVATE
        TinyRayTracingApp.cu)

target_include_directories(Tiny-Ray-Tracing-Lib PUBLIC Geometry)

add_subdirectory(Editor)
add_subdirectory(GUI)
add_subdirectory(Input)
add_subdirectory(Materials)
add_subdirectory(Models)
add_subdirectory(Objects)
add_subdirectory(RenderEngine)
add_subdirectory(Scene)
add_subdirectory(Shaders)
add_subdirectory(Utils)