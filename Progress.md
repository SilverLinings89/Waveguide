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

The solution with HSIE BC is now 0. Rerunning with PML to check if thats the case for PML, too. It is ,because I deactivated th preparation of the rhs. Implementing new version to compute rhs with correct norm. The steps for this are as follows: Project the exact solution to the domain boundary (b_id=4 and index_in_z_direction=0) >> vmult the resulting vector with the system matrix >> project 0 to the incoming interface (b_id=4 and index_in_z_direction=0). The resulting vector represents the incoming field for a jump coupling.

The function has been implemented with the name compute_rhs_representation_of_incoming_wave(). It updated the vector rhs of a Hierarchical Problem type. It should work on Local and NonLocal Problems. The implementation now runs. There were some difficulties about vector compression but they are fixed. 

# Monday, 21th of june

Implementing vmult on Inner domain and precomputation of the local matrix. This is a matrix unaware of boundary conditions which is computet entirely locally. There might be work required later on to compute this precisely for higher order sweeping because then the input interface is distributed to multiple cores.

# Monday, 5th of july

Currently the computation of the solution doesnt work at all. The scale is completely off and the solution builds up nummerical errors added on top of the solution with increasing z. My goal for today is to fix the amplitude of the exact solution. I will first implement a parameter to switch to a direct solver in the solve function. That way I don't always have to recompile and comment / un-comment code.

Other points to fix today: Fix the output to only diplay once fot the messages where it makes sense. Fix the crashes in the output routines. Also make PML Domain outputs compatible with inner outputs (requires Exact solution field). Then add pvtu file.

Added parameter for logging level. Fixed a wrong implementation in print infos function to determine visibility for the ...One versions of the logging level. Bug fixed. Default Logging Level was too low.

Making PML and inner domains compatible next. To make this so, I have to add an exact solution  output to the pml domain output. Done. Now adding pvtu file in non-local problem. The pvtu should contain all inner domains on the current level as well as the PML domain for PML computations. For this implementation, a string return value has been added to all output_result functions which is the filename of the files written. Compiles. Runs.

The implementation has 2 errors: 
- There are no data objects visible, because the dof_output generating the pvtu doesnt know of any.
- The files cannot be found because the paths are wrong.

# Tuesday, 6th of july

The output files are working now. The next step will be to fix the directly solved, non-local problem to the solution it should have. The current problems are:

- The amplitude is too high in the interior. 
- Something weird is happening on the input interface. The coupling seems broken.

There is an interesting solution to this: The coupling interface dofs have 2 values - the input field and the scattered field. The idea is to only compute the scattered field - i.e subtract the input field from the computed field on the input boundary. I can facilitate this with an inhomogenous constraint on the interface because the dof actually exists twice.

# Wednesday, 7th of july

Today I will implement the dof surface split for the surface dofs. There were some issues in doing so. I will continue tomorrow.

# Thursday, 8th of july

To implement the surface split, all that is required is an adaptation of the make_constraints function. I need an aditional one called make_inhomogenous constraints that adds the inhomogeneities. Because they would conflict I have decided to instead add an argument to the make_surface_constrints function of type bool if inhomogeneities should be built. Otherwise I would have to search through the existing contraints and update some lines. Merging is not possible because the constraint has to be x_inner = x_outer + inhom. Merging would give me two entries. x_inner = inhom and x_inner = x_outer which lead to a different result.

Introduced the new Type SignalCouplingMethod and added a member of this type to the Parameters Object. This will help me keep parallel implementations of different coupling methods. Since constraints are always formulated as a constraint on the dof with the lower index, I can extend the function that makes the coupling to also add an inhomogeneity to the line (otherwise I would have to change the sign).

Added an implementation in PMLSurface.cpp make_surface_contraints. Testing it. I temporarily have to disable the compute_rhs_representation_of_incoming_wave for this.

# Friday, 9th of july

Copied the implementation to the HSIE implementation as well. Also added an implementation of Dirichlet boundary values in the Hierarchical Problem class. I also switched the negative signs on the z component in the exact solution. Switched the sign of the interface jump values. This only changes the solutions sign basically. As expected. Trying out some solution steps and sweeping details with Dirichlet boundary data.

# Monday, 12th of july

Frank Hettlich wrote me an Email, that my presentation of domain derivatives looked good to him. I had wondered about his oppinion on this topic since he is a researcher in the field.
I will now focus on fixing the sweeping preconditioner by utilizing Dirichlet boundary data first. I will ignore the complex conjugate and use PML boundary conditions since they produce the right amplitude for the signal. Dirichlet works nicely here. Performing a run (n° 82) with PML, Dirichlet data and a direct solver. Observations: 
- The code is currently somewhat slow in assembly. This might be due to the compute intensive function for the assembly of the off diagonal block for vmult.
- It is also due to a lack of output in the direct solver call in NonLocalProblem::solve() which creates the impression that assembly is still runing even though solution has started. Will put output into NonLocalProblem::solve() to deal with that. Done.
- The surface of the output geometry seems to be exactly zero for the tangential components.
- Orthogonal components are of order 10^{-2} compared to the input amplitude.
- The real parts align perfectly and the solution looks ideal. 0 in the input PML layer (as it should be) and damped in the output PML.
- The imaginary part overshoots by roughly 10% consitently across the domain. Damping works for it and it has the wrong sign.
- In general, however this computation looks fine.
As a next step I will reactivate sweeping and see what happens.

I found an error in my code: zero_lower_interface_dofs returns a new vector, it doesn't change the vector in place. As a consequence, all calls to it I had made, had no effect. Fixed. Still doesn't converge to 0 after the fix. Will analyze the results for better understanding of what is happening.