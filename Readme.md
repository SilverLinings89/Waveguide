# The 3D Maxwell Project

## Setup

Need Petsc with complex double as numbers. Also Mumps.

Have to delete the content of the function project_matrix_free in vector_tools_project.templates.h in DealInstall/include/dealii/numerics/

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
| Sector     | A sector is an expression used in shape modeling and refers to a subset of the area between the connectors. |
| Layer      | used in the implementation and represents the part of the triangulation owned by one Process.  |
| Connector  | The structure in the two halfspaces towards z->\infty and z->-\infty    |
| Space Transformation | The space transformation always also includes the computation of PML or other ABCs. it does not however involve the multiplication with the actual material property. This can be done at the point of usage. |
| ... | ... |

### Levels in Hierarchical Sweeping

There are 2 level types: global and local.

#### Global

The global level describes how many hierarchical levels there are in total.

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

## Side ownership

### Local Problems

All faces are owned except the upper one for the lowes sweeping direction.
For global_rank 3: We sweep x in lowest order so the interface to the -x-direction is not owned.
For global_rank 2: We sweep y in lowest order so the interface to the -y-direction is not owned.
For global_rank 1: We sweep only in the z direction so the interface to the -z-direction is not owned.
The process with rank 0 in the lowest sweeping rank owns that interface (since there is no neighbor).

### Hierarchical problems

0 = -x -> only owned if rank in x = 0;
1 = +x -> always owned
2 = -y -> only owned if rank in y = 0 or for global_rank - rank > 1 (i.e. x-sweeps);
3 =  y -> always owned
4 = -z -> only if rank in z = 0 or for global_rank - rank > 0 (i.e. in y- or x-sweeps);
5 =  z -> always owned

## Edge ownership

Edges are interceptions between two sides. The weaker rule holds. So if side a is owned by side b is not, the edge between a and b is not owned.

## Precomputation

In order to not have to check for every edge, I will precompute ownership based on indices and level. I can compute if I own an edge based on
- global level
- current level
- the boundary id the surface is associated with
- the boundary of the edge
Since global level and the boundary id of the surface are fixed for one object, this is a matrix with (global-level+1)x(6) entries. If I store that data into an array edge_ownership_by_level_and_id[][] I can then simply compute
``` 
is_owned_on_level =  edge_ownership_by_level_and_id[level][surface_b_id] && edge_ownership_by_level_and_id[level][edge_b_id];
```


## Thanks

My thanks go to the CRC 1173 which is funding my research.
