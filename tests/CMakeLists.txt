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

# All .cpp files of this folder belong to the unit tests; a new test file is picked up automatically.
file(GLOB WAVEGUIDE_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(waveguide_tests ${WAVEGUIDE_TEST_SOURCES})
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
  # The tests of the mode solver need numpy and scipy. Use the first interpreter that has both: the one CMake
  # found, python3 of the PATH, or the one given with -DWAVEGUIDE_TEST_PYTHON=<path>. Without one these tests are
  # reported as skipped.
  if(NOT WAVEGUIDE_TEST_PYTHON)
    find_program(_python_of_path python3)
    set(_candidate_found ${Python3_EXECUTABLE})
    foreach(_candidate ${Python3_EXECUTABLE} ${_python_of_path})
      execute_process(COMMAND ${_candidate} -c "import numpy, scipy" RESULT_VARIABLE _import_failed OUTPUT_QUIET ERROR_QUIET)
      if(NOT _import_failed)
        set(_candidate_found ${_candidate})
        break()
      endif()
    endforeach()
    set(WAVEGUIDE_TEST_PYTHON ${_candidate_found} CACHE FILEPATH "Python interpreter for the tests (with numpy and scipy for the mode solver)")
  endif()
  message(STATUS "Python for the tests: ${WAVEGUIDE_TEST_PYTHON}")
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
    run_report
    optimization_recovers_straight_waveguide
    mode_solver_circular_waveguide
    mode_solver_file_cross_section
  )
  foreach(_test ${_integration_tests})
    add_test(NAME integration_${_test}
      COMMAND ${WAVEGUIDE_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_integration_test.py
        --main $<TARGET_FILE:main> --mpiexec ${DEAL_II_MPIEXEC} ${_test})
    set_tests_properties(integration_${_test} PROPERTIES
      LABELS "integration" TIMEOUT 1800 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
  endforeach()
  # The web frontend: configuration handling and parsing of the solver outputs (does not start the solver).
  add_test(NAME frontend_server
    COMMAND ${WAVEGUIDE_TEST_PYTHON} ${CMAKE_SOURCE_DIR}/frontend/test_server.py)
  set_tests_properties(frontend_server PROPERTIES LABELS "unit")
  # The mode solver (Python package modesolver): analytical and numerical modes; skipped without numpy and scipy.
  add_test(NAME mode_solver
    COMMAND ${WAVEGUIDE_TEST_PYTHON} ${CMAKE_SOURCE_DIR}/modesolver/test_modesolver.py)
  set_tests_properties(mode_solver PROPERTIES LABELS "unit" SKIP_RETURN_CODE 77)
else()
  message(STATUS "Python 3 was not found. The integration tests are disabled.")
endif()
