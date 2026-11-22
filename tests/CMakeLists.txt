# GoogleTest: use an installed version if there is one, otherwise download it.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(waveguide_tests
  main.cpp
  BoundaryMethodTests.cpp
  ExactSolutionTests.cpp
  HardyPolynomialTests.cpp
  MeshTests.cpp
  ShapeFunctionTests.cpp
  BasicFem.cpp
)
deal_ii_setup_target(waveguide_tests)
target_link_libraries(waveguide_tests Code GTest::gtest)

# Unit tests (single process).
add_test(NAME unit_tests
  COMMAND waveguide_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

set_tests_properties(unit_tests PROPERTIES LABELS "unit")
