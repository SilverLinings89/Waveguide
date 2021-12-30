SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the name of the project and target:
SET(TARGET "WaveguideProblem")
SET(DEAL_II_DIR "/home/kraft/libs")
SET(VERSION "")
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)
project(waveguide)

FIND_PACKAGE(deal.II 9.2 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ ../../libs $ENV{DEAL_II_DIR}
        )
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path. You need Dealii 9.2 at least."
            )
ENDIF()

enable_testing()

set (CMAKE_CXX_STANDARD 14)
set (gtest_force_shared_crt ON CACHE BOOL "MSVC defaults to shared CRT" FORCE)

add_subdirectory(third_party/googletest)
target_compile_definitions(gtest
    PUBLIC
        GTEST_LANG_CXX20
        GTEST_HAS_TR1_TUPLE=0
)


DEAL_II_INITIALIZE_CACHED_VARIABLES()
ADD_SUBDIRECTORY(Code)
ADD_SUBDIRECTORY(Main)

add_executable(waveguide_tests)
target_sources(waveguide_tests
    PRIVATE
        tests/BoundaryMethodTests.cpp
        tests/ExactSolutionTests.cpp
        tests/HardyPolynomialTests.cpp
        tests/MeshTests.cpp
)
set_target_properties(waveguide_tests PROPERTIES
    COMPILE_FLAGS "${cxx_strict}"
)
DEAL_II_SETUP_TARGET(waveguide_tests)
target_link_libraries(waveguide_tests Code gtest gtest_main)
add_test(waveguide_tests waveguide_tests)

