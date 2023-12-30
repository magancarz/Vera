cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
project(Vera LANGUAGES CXX CUDA)

add_library(Vera-Lib STATIC Source/Vera.cu)

add_subdirectory(Source)
target_include_directories(Vera-Lib PUBLIC Source)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})

add_subdirectory(libs/glm)
target_link_libraries(Vera-Lib PUBLIC glm::glm)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(libs/GLFW)
target_link_libraries(Vera-Lib PUBLIC glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(Vera-Lib PUBLIC OpenGL::GL)

add_subdirectory(libs/imgui)
target_include_directories(Vera-Lib PUBLIC libs/imgui)
target_include_directories(Vera-Lib PUBLIC libs)

find_package(CUDAToolkit 12.2 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
target_link_libraries(Vera-Lib PUBLIC CUDA::cudart)
set_target_properties(Vera-Lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_XX -O3)
include_directories(libs/cuda)

add_executable(Vera Source/main.cu)
target_link_libraries(Vera Vera-Lib)
set_target_properties(Vera PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(CTest)
add_subdirectory(libs/gtest)

if(BUILD_TESTING)
	add_executable(Tests Tests/TestUtils.cu)
	target_link_libraries(Tests PRIVATE GTest::gtest_main)
	target_link_libraries(Tests PRIVATE Vera-Lib)
	set_target_properties(Tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	target_include_directories(Tests PRIVATE Tests)
	include(GoogleTest)
	gtest_discover_tests(Tests)
	add_subdirectory(Tests)
endif()

set_target_properties(
    Vera PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")