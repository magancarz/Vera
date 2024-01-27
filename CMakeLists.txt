cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
project(Vera LANGUAGES CXX)

add_library(Vera-Lib STATIC Source/Vera.cpp)

add_subdirectory(Source)
target_include_directories(Vera-Lib PUBLIC Source)

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})
link_libraries(${Vulkan_LIBRARIES})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(libs/GLFW)
target_link_libraries(Vera-Lib PUBLIC glfw)

add_subdirectory(libs/imgui)
target_include_directories(Vera-Lib PUBLIC libs/imgui)
target_include_directories(Vera-Lib PUBLIC libs)

add_executable(Vera Source/main.cpp)
target_link_libraries(Vera Vera-Lib)

#include(CTest)
#add_subdirectory(libs/gtest)
#
#if(BUILD_TESTING)
#	add_executable(Tests Tests/TestUtils.cu)
#	target_link_libraries(Tests PRIVATE GTest::gtest_main)
#	target_link_libraries(Tests PRIVATE Vera-Lib)
#	target_include_directories(Tests PRIVATE Tests)
#	include(GoogleTest)
#	gtest_discover_tests(Tests)
#	add_subdirectory(Tests)
#endif()