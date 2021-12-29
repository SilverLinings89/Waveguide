# The 3D Maxwell Project

## Setup

Software dependencies: 
- Dealii 9.3.
- Petsc with complex doubles as number type.
- Mumps (as a dealii dependency)


Have to delete the content of the function project_matrix_free in vector_tools_project.templates.h in DealInstall/include/deal.II/numerics/

Before the first run, one dependency has to be fetched. To do so, navigate a terminal to the main folder of the repo. Then run the commands `git submodule init` and `git submodule update`. This will clone the gtest framework into the third_party folder. Afterwards, create a folder called `build` in the main folder, enter it and run cmake. I.e.
```
git submodule init
git submodule update
mkdir build
cd build
cmake ..
```

## Introduction

This code was developed by Pascal Kraft in an effort to generate a fast code to compute optimal shapes of 3D-waveguides based on a solution of the full Maxwell-Problem without simplifications based on non-physical assumptions. 

Core elements are a refined adjoint based optimization scheme and a sweeping preconditioner. 

This is a work in progress and not currently ready for general usage. Please refer to pascal.kraft@kit.edu for more information.

There are currently works in the branch complex-numbers that might need to be pulle.

## Naming convenctions

| Expression |Meaning     |
|------------|------------------------------------------------------------------------|
| Sector     | A sector is an expression used in shape modeling and refers to a subdomain between the connectors. |
| Layer      | used in the implementation and represents the part of the triangulation owned by one Process.  |
| Connector  | The structure in the two halfspaces towards z->\infty and z->-\infty  outside of the computational domain.  |
| ... | ... |

### Levels in Hierarchical Sweeping

There are 2 level types: global and local.

#### Global

The global level describes how many hierarchical levels there are in total.

- 0: purely local computation, no sweeping.
- 1: there is only sweeping in the z direction.
- 2: there is sweeping in the z direction and the blocks are solved by sweeping in the y direction
- 3: there is sweeping in the z direction and the blocks are solved by sweeping in the y direction. The blocks required in the y-sweep are solved by sweeping the x-direction.

#### Local

It makes a difference for which of the sweeps a matrix is assebled and what the global sweeping level is. The local level describes for which scenario the block is assembled. It counts from 0 up. 0 means that we assemble a local block, i.e. a direct solver for the local problem with sweeping in only one direction. 
If local is 0 and global is 1, there is only sweeping in the z-direction. That means the local block is assembled by cutting off the x- and y- directions.
If local is 0 and global is 2, the lowest order sweeping is in the y-direction, so the direct solver is assembled for a domain with boundary conditions in x- and z-direction but sweeping in y (similar for local 0 and global 3, only for x).
If local is 1 or higher, the block is not assembled for a direct solver but a matrix for GMRES is built. This means that not all surfaces are either boundaries with HSIE or sweeping block interfaces (Dirichlet boundaries), they can now be internal.

## Rotations for surface extraction

For boundary id:
| b_{id} | x | y |
| --- | --- | ---|
| 0 | z | y |
| 1 | z | y |
| 2 | x | z |
| 3 | x | -z |
| 4 | -x | y |
| 5 | x | y |

## Predefined waveguide shapes

This code comes with some example cases called the waveguide hump examples. These examples are available as hdf5 and obj files in the folder Modes -> PWB_Bond_Trajectories. The first three cases are numbered 0,1 and 2 and then continue on at 40, 41, ... , 72. In total there are 3 + 33 cases, so 36. To run these, a parameter in the case file can be used. Be carefull, however, because these numbers start at 0 and count to 35. The table below lists how these numbers correlate: 
| PWB Bond index | Predefined case number | Approximate signal loss in % | Approximate signal loss in db |
| --- | --- | --- | --- |
| 0 | 0 |  |  |
| 1 | 1 |  |  |
| 2 | 2 |  |  |
| 40 | 3 |  |  |
| 41 | 4 |  |  |
| 42 | 5 |  |  |
| 43 | 6 |  |  |
| 44 | 7 |  |  |
| 45 | 8 |  |  |
| 46 | 9 |  |  |
| 47 | 10 |  |  |
| 48 | 11 |  |  |
| 49 | 12 |  |  |
| 50 | 13 |  |  |
| 51 | 14 |  |  |
| 52 | 15 |  |  |
| 53 | 16 |  |  |
| 54 | 17 |  |  |
| 55 | 18 |  |  |
| 56 | 19 |  |  |
| 57 | 20 |  |  |
| 58 | 21 |  |  |
| 59 | 22 |  |  |
| 60 | 23 |  |  |
| 61 | 24 |  |  |
| 62 | 25 |  |  |
| 63 | 26 |  |  |
| 64 | 27 |  |  |
| 65 | 28 |  |  |
| 66 | 29 |  |  |
| 67 | 30 |  |  |
| 68 | 31 |  |  |
| 69 | 32 |  |  |
| 70 | 33 |  |  |
| 71 | 34 |  |  |
| 72 | 35 |  |  |


## Thanks

My thanks go to the CRC 1173 which is funding my research. I also thank the team behind the deal.II library that has been the basis for my implementations.
