currentpath=`pwd`
dealversion="9.3.0"
petscversion="3.13.4"
sudo apt-get update
sudo apt-get install bison flex openmpi-bin openmpi-common libopenmpi-dev libopenblas-base libopenblas-dev libboost1.71-dev libtbb-dev libtbb2 libxt-dev g++ gcc cmake-curses-gui python-dev gfortran cmake libblas-dev liblapack-dev python valgrind zlib1g zlib1g-dev libgmp-dev

mkdir dealii
cd dealii
wget https://github.com/dealii/dealii/releases/download/v${dealversion}/dealii-${dealversion}.tar.gz
tar -xf dealii-${dealversion}.tar.gz
mv dealii-${dealversion} dealii-source
cd ..
mkdir p4est
cd p4est
wget http://p4est.github.io/release/p4est-2.2.tar.gz
wget https://www.dealii.org/current/external-libs/p4est-setup.sh
cd ..
mkdir suitesparse
cd suitesparse
wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.4.0.tar.gz
tar -xf SuiteSparse-5.4.0.tar.gz
cd ..
mkdir petsc
cd petsc
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-${petscversion}.tar.gz
tar -xzf petsc-${petscversion}.tar.gz
mkdir $HOME/libs
mv petsc-${petscversion} $HOME/libs/
cd ..
mkdir gmsh
cd gmsh
wget http://gmsh.info/bin/Linux/gmsh-git-Linux64.tgz
tar -xf gmsh-git-Linux64.tgz
cd ..
mkdir symengine
cd symengine
wget https://github.com/symengine/symengine/releases/download/v0.6.0/symengine-0.6.0.tar.gz
tar -xf symengine-0.6.0.tar.gz
mv symengine-0.6.0 symengine
cd ..
