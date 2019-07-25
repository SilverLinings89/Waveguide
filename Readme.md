# The 3D Maxwell Project

This code was developed by Pascal Kraft in an effort to generate a fast code to compute optimal shapes of 3D-waveguides based on a solution of the full Maxwell-Problem without simplifications based on non-physical assumptions. 

Core elements are a refined adjoint based optimization scheme and a sweeping preconditioner. 

This is a work in progress and not currently ready for general usage. Please refer to pascal.kraft@kit.edu for more information.

There are currently works in the branch complex-numbers that might need to be pulle.

# Naming convenctions

| Expression |Meaning     |
|------------|------------------------------------------------------------------------|
| Sector     | A sector is an expression used in shape modeling and refers to a subset of the area between the connectors. |
| Layer      | used in the implementation and represents the part of the triangulation owned by one Process.  |
| Connector  | The structure in the two halfspaces towards z->\infty and z->-\infty    |
| Space Transformation | The space transformation always also includes the computation of PML or other ABCs. it does not however involve the multiplication with the actual material property. This can be done at the point of usage. |
| ... | ... |

# Thanks

My thanks go to the CRC 1173 which is funding my research.
