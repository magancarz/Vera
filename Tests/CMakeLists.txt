target_sources(Vera-Tests PRIVATE
        Environment.cpp
        main.cpp
        TestLogger.cpp
        TestUtils.cpp)

add_subdirectory(E2E)
add_subdirectory(IntegrationTests)
add_subdirectory(Mocks)
add_subdirectory(UnitTests)