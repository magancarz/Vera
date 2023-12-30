target_sources(Vera-Lib PRIVATE
        Ray.h
        RayTracedImage.cpp
        RayTracer.cu
        RayTracerCamera.cu
        RayTracerConfig.h
        ScatterRecord.h)

add_subdirectory(IntersectionAccelerators)
add_subdirectory(PDF)
add_subdirectory(Shapes)