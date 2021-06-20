# Thursday, 17th of june

- Fix convergence
- Everything else is not that important
- Dont get distracted
- Write down my progress for document

## Steps

1. Document the results of various versions
2. Figure out implementation for expected result

## Thoughts

The current issue appears to be that the interface degrees of freedom are not handeled correctly. As a consequence, the interfaces introduce disturbances in the system that lead to artificial solutions and non-convergence of the scheme.

There are two ways to make progress: 
- Implementing possible solutions and mapping outcome to implementation and eventually predict the right implementation by solutions components required to acchieve the right solution.
- Derive the correct implementation from the mathematical basis. This appears to be more accurate and while it seems more tedious I will go this way.

## The right implementation

The paper expects the "upper" dofs not to be included in the locally owned set - this however should be interchangeable. The important point is not to have them treated twice. Problems arise from the constraint matrix structure in Dealii that has to be used to assemble the matrix and couple the local contributions together. 
It currently seems safe to assume that the implementation of the boundary constraints is correct but for the details of local ownership. I should transition to a system where the lower process gets the matrix line containing the contribution such that there is no misunderstanding about which process stores a line.
Explanation: If I always couple own-to-other, i.e couple the line of the own surface dof to the column of the other dof, I don't get the same coplings. They are logically the same (it doesnt matter if I write x_1 = x_3 or x_3 = x_1, but it might make a difference for storage when x_1 is locally owned and x_3 is not and vice versa.) Implementing this now. Done. Check result.

I configured a default build action that can be triggered by ctrl + shift + b.

First version had distribute from reading the vector but no set_zero before updating the local solution. The computed result (run135) shows a small amplitude in domain 0, bad interface coupling and cell-layer distorted, wrong results in doamins 2 and 3. Also: The residual is very large iniatially, which likely has to do with the erroneous behaviour in domains 2 and 3. I activated set_zero and reran. The solution looks identical (run136).

I assume that adding the correct constraints might have highlighted errors I made elsewhere and will therefore attemt to recreate an implementation of exactly the algorithm stated in the paper.

Added set_zero and distribute to all library calls for solvers and vmult.

Next step: I will reintroduce norm outputs to see where the high solution arises. Added norm output to all important lines in apply sweep.

Evaluation: Zeroes untill end of down sweep. Norm 23 for proc 0. The direct solves and vmult seem unstable. Adding a direct solve for the parent system to see if the solution of the global problem is correct. Amplitude of the exact solution is too large. Adding set_zero and distribute for direct solver run 141.

Since the earlier issues was adressed, there is no more phase shift for PML so I can output the correct solution instead of the conjugate for comparison to the exaxt solution. I will keep using a direct solver until the solution is correct.

In Inner Domain: Assemble functions where missing a call to compress(add) which is now added.

A better way to compute the rhs would be to project the exact solution onto the domain for a vector u_inc. Then multiply this vector with the system matrix and to set the components of rhs, where u_inc != 0 to 0.

Short comparison to check if HSIE is still working. This will be run 144. run 143 was the run with a direct solver and PML. Run crashed.

Made global indices of HSIE dofs actually global. Dof counter counts the number of dofs and global indices are the numbers from first_own_index up to first_own_index + dof_counter.

Currently working on fixing an issue in HSIE. On higher levels, the surface dofs are not flagged properly. Issue occurs on all surfaces for higher level. Dof Handlers dont have this issue. Might be related to dof indexing.

Bug fixed. The issue was that user flags were set on the surface mesh. They are now being cleared at the beginning of init. This is an issue because the surface tria is being reused on all levels.

# Friday, 18th of june

While fixing HSIE for the higher levels I ran into errors. The current error is, that the MPI exchange function for sparsity patterns doesn't receive data. I am searching for differences between PML and HSIE to find the bug.

Removed some old logging from the constraint generation. Added some new output in sparsity functions. As expected the problem occurs for neighbour surfaces.

Corner cell dof vector computation is not implemented. Fixing. The function is now implemented. Extended the function to also work when one of the boundary Ids is the b_id value of the surface object.

I had to reload the repo because git broke it again. Cloned from the repo and saved local changes.

# Sunday, 20th of june

Default run task still works. Assembly still works for PML. DSP for HSIE not working. Changed is_point_at_boundary in HSIESurface to return true if bounday id is either inner or outer, previously it was true for inner, false for outer. This led to the empty arrays. Assembly runs completely. There was an error attempting to compress datastructures. Adding back the MPI Barriers in the assemble function to block race conditions. The matrix->compress operation only terminates for Process 0 and 3, not for 1 and 2.

The issue occurs in the function fill_matrix most likely of the hsie_surface. Added output to make sure that's it. All start their calls (call for surface 5 starts). The error was the missing line "matrix->compress(dealii::VectorOperation::add);" at the end of fill matrix for HSIE surfaces. Running ... Worked. Assembly runs through now. Solve worked, too. Detail: The application no longer crashes in the output. That means the crashes (which were irrelevant so far) occured in the PML output or something related to the output of the solution on the  boundaries.

The solution with HSIE BC is now 0. Rerunning with PML to check if thats the case for PML, too.