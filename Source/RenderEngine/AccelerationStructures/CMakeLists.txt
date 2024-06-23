target_sources(Vera-Lib PRIVATE
        AABB.cpp
        AccelerationStructure.cpp
        BlasBuilder.cpp
        Tlas.cpp)

add_subdirectory(Blas)
add_subdirectory(BVH)
add_subdirectory(Octree)