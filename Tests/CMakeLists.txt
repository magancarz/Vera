target_sources(Vera-Tests PRIVATE
        Environment.cpp
        main.cpp
        TestUtils.cpp)

add_subdirectory(IntegrationTests)
add_subdirectory(Mocks)
add_subdirectory(UnitTests)