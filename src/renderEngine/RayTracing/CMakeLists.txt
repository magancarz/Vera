target_sources(Tiny-Ray-Tracing-Lib PRIVATE
        Ray.h
        ImageBlock.h
        RayTracedImage.cpp
        RayTracer.cu
        HybridRayTracer.cu
        RayTracerCamera.cu
        RayTracerConfig.h
        ScatterRecord.h)

add_subdirectory(IntersectionAccelerators)
add_subdirectory(PDF)
add_subdirectory(Shapes)