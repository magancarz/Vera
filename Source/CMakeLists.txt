target_sources(Vera-Lib PRIVATE
        Vera.cpp)

target_include_directories(Vera-Lib PUBLIC Geometry)

add_subdirectory(Editor)
add_subdirectory(Input)
add_subdirectory(Objects)
add_subdirectory(Project)
add_subdirectory(RenderEngine)
add_subdirectory(Utils)
add_subdirectory(World)