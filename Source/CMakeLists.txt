target_sources(Vera-Lib PRIVATE
        Vera.cpp)

target_include_directories(Vera-Lib PUBLIC Geometry)

add_subdirectory(Editor)
add_subdirectory(GUI)
add_subdirectory(Input)
add_subdirectory(RenderEngine)
add_subdirectory(Utils)