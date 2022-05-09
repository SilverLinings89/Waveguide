currentpath=`pwd`
dealversion="9.2.0"
pjobs="4"
petscversion="3.13.4"

cd ${currentpath}
cd p4est
chmod u+x p4est-setup.sh
./p4est-setup.sh p4est-2.2.tar.gz $HOME/libs

# Install Umfpack and metis

cd ${currentpath}
cd suitesparse
cd SuiteSparse
make -j ${pjobs}
make install INSTALL=$HOME/libs

# Install Gmsh

cd ${currentpath}
cp gmsh/gmsh-git-Linux64/* $HOME/libs/

#install Symengine

cd ${currentpath}
cd symengine
cd symengine
mkdir build
cd build
CXXFLAGS=-fPIC cmake -DCMAKE_INSTALL_PREFIX:PATH="$HOME/libs" ..
make
make install

#install PETSC

export PETSC_DIR=$HOME/libs/petsc-${petscversion}
cd ${PETSC_DIR}/
export PETSC_ARCH=x86_64
./configure --with-shared=1 --with-x=0 --with-mpi=1 --download-hypre=1 --with-precision=double --with-scalar-type=complex --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch
make PETSC_DIR=$HOME/libs/petsc-${petscversion} PETSC_ARCH=x86_64 all

# Install deal ...

cd ${currentpath}
cd dealii
mkdir dealbuild
cd dealbuild
cp ../../deal_conf ./deal_conf
chmod a+x ./deal_conf
./deal_conf
make -j ${pjobs}
make install
