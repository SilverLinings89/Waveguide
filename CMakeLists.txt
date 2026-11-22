cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# deal.II is located via -DDEAL_II_DIR=<prefix>, the environment variable
# DEAL_II_DIR or one of the hints below.
find_package(deal.II 9.6 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR} ../deps/install/dealii ~/libs
)

foreach(_feature MPI PETSC SYMENGINE COMPLEX_VALUES)
  if(NOT DEAL_II_WITH_${_feature})
    message(FATAL_ERROR
      "This code requires a deal.II that was configured with DEAL_II_WITH_${_feature}=ON.\n"
      "The deal.II found in ${DEAL_II_PATH} was not. See Readme.md (section Installation).")
  endif()
endforeach()
if(NOT DEAL_II_PETSC_WITH_COMPLEX)
  message(FATAL_ERROR
    "PETSc has to be configured with --with-scalar-type=complex (and MUMPS). See Readme.md.")
endif()

deal_ii_initialize_cached_variables()
project(waveguide CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Root of the repository. Used to locate Modes/ and Parameters/ independently of
# the directory the executable is started from. Can be overridden at run time
# with the environment variable WAVEGUIDE_ROOT.
add_compile_definitions(WAVEGUIDE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_subdirectory(Code)
add_subdirectory(Main)

option(WAVEGUIDE_BUILD_TESTS "Build the unit and integration tests" ON)
if(WAVEGUIDE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
