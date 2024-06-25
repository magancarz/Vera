cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
project(Vera LANGUAGES C CXX)

add_library(Vera-Lib STATIC Source/Vera.cpp)

add_subdirectory(Source)
target_include_directories(Vera-Lib PUBLIC Source)
target_include_directories(Vera-Lib PUBLIC ThirdParty)

find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
include_directories(${Vulkan_INCLUDE_DIRS})
link_libraries(${Vulkan_LIBRARIES})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ThirdParty/GLFW)
target_link_libraries(Vera-Lib PUBLIC glfw)

add_subdirectory(ThirdParty/imgui)
target_include_directories(Vera-Lib PUBLIC ThirdParty/imgui)

add_subdirectory(ThirdParty/GLM)
target_link_libraries(Vera-Lib PUBLIC glm::glm)

add_subdirectory(ThirdParty/VMA)
target_link_libraries(Vera-Lib PUBLIC GPUOpen::VulkanMemoryAllocator)

add_executable(Vera main.cpp)
target_link_libraries(Vera Vera-Lib)
set_target_properties(Vera PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
file(GLOB SHADERS
    ${SHADER_SOURCE_DIR}/*.vert
    ${SHADER_SOURCE_DIR}/*.frag
    ${SHADER_SOURCE_DIR}/*.rchit
    ${SHADER_SOURCE_DIR}/*.rahit
    ${SHADER_SOURCE_DIR}/*.rgen
    ${SHADER_SOURCE_DIR}/*.rmiss
    ${SHADER_SOURCE_DIR}/*.comp)

add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
    OUTPUT ${SHADER_BINARY_DIR}
    COMMENT "Creating ${SHADER_BINARY_DIR}"
)

foreach (source IN LISTS SHADERS)
	get_filename_component(FILENAME ${source} NAME)
	add_custom_command(
        COMMAND ${glslc_executable} -o ${SHADER_BINARY_DIR}/${FILENAME}.spv ${source} --target-env=vulkan1.2
        OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
        DEPENDS ${source} ${SHADER_BINARY_DIR}
        COMMENT "Compiling ${FILENAME}"
	)
	list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach ()

add_custom_target(SHADERS ALL DEPENDS ${SPV_SHADERS})
add_dependencies(Vera SHADERS)

include(CTest)
enable_testing()
add_subdirectory(ThirdParty/gtest)

if(BUILD_TESTING)
	file(GLOB_RECURSE SOURCES
        Source/*.cpp
        Source/*.h)
	add_executable(Vera-Tests ${SOURCES})
	target_link_libraries(Vera-Tests PRIVATE Vera-Lib)
	target_link_libraries(Vera-Tests PRIVATE GTest::gtest_main)
	target_link_libraries(Vera-Tests PRIVATE GTest::gmock)
	include(GoogleTest)
	target_include_directories(Vera-Tests PRIVATE Tests)
	add_test(NAME Vera-Tests COMMAND Vera-Tests WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
	message("${CMAKE_SOURCE_DIR}")
	add_subdirectory(Tests)
endif()

option(coverage "Build for coverage report." OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_COMPILER_IS_GNUCXX AND coverage)
    message("Coverage report ENABLED")

    include(cmake/CodeCoverage.cmake)
    append_coverage_compiler_flags()
    setup_target_for_coverage_gcovr_html(NAME Vera-Tests-Coverage
            EXECUTABLE ctest --verbose -C Debug
            EXCLUDE ${CMAKE_SOURCE_DIR}/Tests/ ${CMAKE_SOURCE_DIR}/ThirdParty/)
else ()
    message("Coverage report disabled")
endif ()
