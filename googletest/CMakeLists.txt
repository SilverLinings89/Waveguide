
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.0 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(Google_tests)

add_subdirectory(lib)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(Google_Tests_run ExampleTest.cpp)
DEAL_II_SETUP_TARGET(Google_Tests_run)
target_link_libraries(Google_Tests_run Code)

target_link_libraries(Google_Tests_run gtest gtest_main)