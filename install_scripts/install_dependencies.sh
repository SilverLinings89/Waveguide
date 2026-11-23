#!/bin/bash
# Builds the dependencies of the waveguide solver from source:
#
#   PETSc   (complex scalars, with MUMPS, ScaLAPACK, METIS and ParMETIS)
#   deal.II (with MPI, PETSc, complex values and SymEngine; debug and release libraries)
#
# Usage:   ./install_dependencies.sh [prefix]
# Default: prefix = <repository>/../deps  (this is where CMakeLists.txt looks for deal.II)
#
# Prerequisites that have to be installed with the package manager of the system first:
#   macOS (Homebrew):  brew install cmake ninja open-mpi gcc boost symengine
#   Ubuntu / Debian:   sudo apt install build-essential gfortran cmake ninja-build libopenmpi-dev \
#                        libboost-all-dev libsymengine-dev libblas-dev liblapack-dev python3
#
# On a machine without MPI and without root access (a cluster that only offers a compiler and CMake), set
#   SELF_CONTAINED=1 ./install_dependencies.sh [prefix]
# The script then also builds MPICH and OpenBLAS (through PETSc), GMP and SymEngine, and uses make if there is
# no ninja. Only gcc, g++, gfortran, cmake, make, curl and python3 are needed. This mode is switched on
# automatically if no mpicc is found. Afterwards put <prefix>/install/petsc/bin into PATH to get mpicc and mpiexec.
#
# The versions below are the ones the code was last tested with (November 2026).
set -e

PETSC_VERSION=${PETSC_VERSION:-3.26.1}
DEALII_VERSION=${DEALII_VERSION:-9.8.0}
GMP_VERSION=${GMP_VERSION:-6.3.0}
SYMENGINE_VERSION=${SYMENGINE_VERSION:-0.15.0}
JOBS=${JOBS:-8}
if [ -z "$SELF_CONTAINED" ]; then
  if command -v mpicc > /dev/null 2>&1; then SELF_CONTAINED=0; else SELF_CONTAINED=1; fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${1:-$SCRIPT_DIR/../../deps}"
mkdir -p "$PREFIX/src" "$PREFIX/install" "$PREFIX/build/dealii"
PREFIX="$(cd "$PREFIX" && pwd)"
# A prefix that was built in the self-contained mode stays in this mode (its mpicc may be in PATH by now).
if [ -d "$PREFIX/install/symengine" ] || [ -x "$PREFIX/install/petsc/bin/mpicc" ]; then SELF_CONTAINED=1; fi

if [ "$SELF_CONTAINED" = 1 ]; then
  REQUIRED_TOOLS="cmake make gcc g++ gfortran curl python3"
  PETSC_COMPILERS="--with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-openblas"
  export PATH="$PREFIX/install/petsc/bin:$PATH"
else
  REQUIRED_TOOLS="cmake ninja mpicc mpicxx mpifort"
  PETSC_COMPILERS="--with-cc=mpicc --with-cxx=mpicxx --with-fc=mpifort"
fi
for tool in $REQUIRED_TOOLS; do
  command -v $tool > /dev/null 2>&1 || { echo "Required tool '$tool' was not found. See the header of this script."; exit 1; }
done
if command -v ninja > /dev/null 2>&1; then
  GENERATOR="Ninja"
else
  GENERATOR="Unix Makefiles"
fi

# ----------------------------------------------------------------------------------------------- PETSc
if [ ! -f "$PREFIX/install/petsc/lib/petsc/conf/petscvariables" ]; then
  cd "$PREFIX/src"
  if [ ! -d petsc-$PETSC_VERSION ]; then
    curl -fsSL -o petsc-$PETSC_VERSION.tar.gz \
      https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-$PETSC_VERSION.tar.gz
    tar -xzf petsc-$PETSC_VERSION.tar.gz
  fi
  cd petsc-$PETSC_VERSION
  unset PETSC_DIR PETSC_ARCH
  ./configure --prefix="$PREFIX/install/petsc" \
    $PETSC_COMPILERS \
    --with-scalar-type=complex --with-debugging=0 --with-x=0 --with-shared-libraries=1 \
    COPTFLAGS="-O2" CXXOPTFLAGS="-O2" FOPTFLAGS="-O2" \
    --download-mumps --download-scalapack --download-metis --download-parmetis
  PETSC_ARCH_NAME=$(ls -d arch-* | head -1)
  make PETSC_DIR="$PWD" PETSC_ARCH=$PETSC_ARCH_NAME all
  make PETSC_DIR="$PWD" PETSC_ARCH=$PETSC_ARCH_NAME install
fi

# ------------------------------------------------------------------------------------- GMP and SymEngine
SYMENGINE_ARGS=""
if [ "$SELF_CONTAINED" = 1 ]; then
  if [ ! -f "$PREFIX/install/gmp/lib/libgmp.so" ] && [ ! -f "$PREFIX/install/gmp/lib/libgmp.dylib" ]; then
    cd "$PREFIX/src"
    if [ ! -d gmp-$GMP_VERSION ]; then
      curl -fsSL -o gmp-$GMP_VERSION.tar.gz https://ftp.gnu.org/gnu/gmp/gmp-$GMP_VERSION.tar.gz \
        || curl -fsSL -o gmp-$GMP_VERSION.tar.gz https://gmplib.org/download/gmp/gmp-$GMP_VERSION.tar.gz
      tar -xzf gmp-$GMP_VERSION.tar.gz
    fi
    cd gmp-$GMP_VERSION
    ./configure --prefix="$PREFIX/install/gmp" --enable-shared --disable-static
    make -j $JOBS
    make install
  fi
  if [ ! -d "$PREFIX/install/symengine/include/symengine" ]; then
    cd "$PREFIX/src"
    if [ ! -d symengine-$SYMENGINE_VERSION ]; then
      curl -fsSL -o symengine-$SYMENGINE_VERSION.tar.gz \
        https://github.com/symengine/symengine/releases/download/v$SYMENGINE_VERSION/symengine-$SYMENGINE_VERSION.tar.gz
      tar -xzf symengine-$SYMENGINE_VERSION.tar.gz
    fi
    cmake -S symengine-$SYMENGINE_VERSION -B "$PREFIX/build/symengine" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX/install/symengine" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$PREFIX/install/gmp" -DCMAKE_INSTALL_RPATH="$PREFIX/install/gmp/lib" \
      -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build "$PREFIX/build/symengine" -j $JOBS
    cmake --install "$PREFIX/build/symengine"
  fi
  SYMENGINE_ARGS="-DSYMENGINE_DIR=$PREFIX/install/symengine"
fi

# --------------------------------------------------------------------------------------------- deal.II
cd "$PREFIX/src"
if [ ! -d dealii-$DEALII_VERSION ]; then
  curl -fsSL -o dealii-$DEALII_VERSION.tar.gz \
    https://github.com/dealii/dealii/releases/download/v$DEALII_VERSION/dealii-$DEALII_VERSION.tar.gz
  tar -xzf dealii-$DEALII_VERSION.tar.gz
fi
# deal.II 9.8.0 does not compile against a complex-valued PETSc without this patch.
for patch in "$SCRIPT_DIR"/patches/dealii-$DEALII_VERSION-*.patch; do
  [ -f "$patch" ] || continue
  if patch -d "$PREFIX/src/dealii-$DEALII_VERSION" -p1 -N --dry-run < "$patch" > /dev/null 2>&1; then
    patch -d "$PREFIX/src/dealii-$DEALII_VERSION" -p1 -N < "$patch"
  fi
done

if [ -z "$SYMENGINE_ARGS" ] && command -v brew > /dev/null 2>&1; then
  SYMENGINE_ARGS="-DSYMENGINE_DIR=$(brew --prefix symengine)"
fi

cd "$PREFIX/build/dealii"
cmake -G "$GENERATOR" "$PREFIX/src/dealii-$DEALII_VERSION" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX/install/dealii" \
  -DCMAKE_BUILD_TYPE=DebugRelease \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DDEAL_II_WITH_MPI=ON \
  -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR="$PREFIX/install/petsc" -DPETSC_ARCH="" \
  -DDEAL_II_WITH_COMPLEX_VALUES=ON \
  -DDEAL_II_WITH_SYMENGINE=ON $SYMENGINE_ARGS \
  -DDEAL_II_WITH_TRILINOS=OFF -DDEAL_II_WITH_P4EST=OFF -DDEAL_II_WITH_HDF5=OFF \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF
cmake --build . -j $JOBS
cmake --install .

# deal.II 9.8.0 does not install two headers of the optimization module (solver_bfgs.h, line_minimization.h)
# when it is built with CMake >= 3.28. Copy every header that was left out.
(cd "$PREFIX/src/dealii-$DEALII_VERSION/include" && find deal.II -name "*.h" | while read -r header; do
  if [ ! -f "$PREFIX/install/dealii/include/$header" ]; then
    mkdir -p "$PREFIX/install/dealii/include/$(dirname "$header")"
    cp "$header" "$PREFIX/install/dealii/include/$header"
  fi
done)

# Homebrew's SymEngine records a versioned LLVM path that disappears with the next LLVM update.
if [ "$SELF_CONTAINED" != 1 ] && command -v brew > /dev/null 2>&1; then
  sed -i.bak -E "s#$(brew --prefix)/Cellar/llvm/[^/;\"]+#$(brew --prefix)/opt/llvm#g" \
    "$PREFIX/install/dealii/lib/cmake/deal.II/deal.IITargets.cmake"
fi

echo
echo "Done. Configure the solver with:"
if [ "$SELF_CONTAINED" = 1 ]; then
  echo "  export PATH=$PREFIX/install/petsc/bin:\$PATH     # mpicc, mpiexec"
fi
echo "  cmake -S <repository> -B <repository>/build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$PREFIX/install/dealii"
