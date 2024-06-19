target_sources(Vera-Lib PRIVATE
        AccelerationStructure.cpp
        BlasBuilder.cpp
        Tlas.cpp)

add_subdirectory(Blas)
add_subdirectory(Octree)