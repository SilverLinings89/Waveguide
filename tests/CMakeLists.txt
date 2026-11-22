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
  ModeAndGeometryTests.cpp
)
deal_ii_setup_target(waveguide_tests)
target_link_libraries(waveguide_tests Code GTest::gtest)

# Unit tests (single process).
add_test(NAME unit_tests
  COMMAND waveguide_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Integration tests. They run the solver with MPI on small problems (2 to 8 processes) and
# compare the results with a direct solve or with the expected physical behavior.
#   ctest -L unit          runs only the fast tests,
#   ctest -L integration   runs the integration tests.
set_tests_properties(unit_tests PROPERTIES LABELS "unit")
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  set(_integration_tests
    straight_waveguide_dirichlet
    straight_waveguide_taper
    mesh_refinement_improves_transmission
    reproducibility
    sweeping_matches_direct_solver
    four_subdomains
    hierarchical_level_2
    hierarchical_level_3
    partition_independence
    block_low_rank_compression
    shape_gradient
    hsie_single_face
    hsie_matches_pml
  )
  foreach(_test ${_integration_tests})
    add_test(NAME integration_${_test}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_integration_test.py
        --main $<TARGET_FILE:main> --mpiexec ${DEAL_II_MPIEXEC} ${_test})
    set_tests_properties(integration_${_test} PROPERTIES
      LABELS "integration" TIMEOUT 1800 RUN_SERIAL TRUE)
  endforeach()
  # The web frontend: configuration handling and parsing of the solver outputs (does not start the solver).
  add_test(NAME frontend_server
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/frontend/test_server.py)
  set_tests_properties(frontend_server PROPERTIES LABELS "unit")
else()
  message(STATUS "Python 3 was not found. The integration tests are disabled.")
endif()
