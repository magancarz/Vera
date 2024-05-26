cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 23)
project(Vera LANGUAGES CXX)

add_library(Vera-Lib STATIC Source/Vera.cpp)

add_subdirectory(Source)
target_include_directories(Vera-Lib PUBLIC Source)

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
target_include_directories(Vera-Lib PUBLIC ThirdParty)

add_executable(Vera main.cpp)
target_link_libraries(Vera Vera-Lib)

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
file(GLOB SHADERS
		${SHADER_SOURCE_DIR}/*.vert
		${SHADER_SOURCE_DIR}/*.frag
		${SHADER_SOURCE_DIR}/*.rchit
		${SHADER_SOURCE_DIR}/*.rahit
		${SHADER_SOURCE_DIR}/*.rgen
		${SHADER_SOURCE_DIR}/*.rmiss)

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
	add_executable(Tests ${SOURCES})
	set_target_properties(Tests PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
	target_link_libraries(Tests PRIVATE GTest::gtest_main)
	target_link_libraries(Tests PRIVATE GTest::gmock)
	target_link_libraries(Tests PRIVATE Vera-Lib)
	include(GoogleTest)
	target_include_directories(Tests PRIVATE Tests)
	add_test(NAME Tests COMMAND Tests)
	add_subdirectory(Tests)
endif()

set_target_properties(Vera PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")