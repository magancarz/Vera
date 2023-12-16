target_sources(Tiny-Ray-Tracing-Lib PRIVATE
        CreateObjectCommand.cpp
        ClearOutlinedObjectsCommand.cpp
        SetObjectToBeOutlinedCommand.cpp
        RefreshRayTracerConfigCommand.cpp
        ToggleLiveRayTracingCommand.cpp
        GenerateRayTracedImagesQueueCommand.cpp
        StopGeneratingImageCommand.cpp
        SaveProjectCommand.cpp
        LoadProjectCommand.cpp
        ChangeProjectNameCommand.cpp
        ChangeRayTracerCommand.cpp)
