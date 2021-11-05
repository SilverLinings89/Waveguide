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

# Monday, 26th of july

Picking up after the vacation. Git status clean. Should maybe automate or at least finally remember git submodule init and git submodule update.
Updated the Readme to list the necessary steps. Build runs fine. My prime focus right now is to fix sweeping. To this end I need to figure out the current issue and so so as fast as possible. In theory, one forward sweep should suffice for the straigt waveguide. I will implement that first.
Overwriting the current Call to GMRES with the new, single sweep implementation. Basic implementation done. Testing. In the first attempt (run2) I see a sign change between processes. Otherwise, the solution looks good. Fixing it by calling subtract_fields on a zero field and the field generated from the trace.

There is still a jump at interfaces. Now that I have disabled constraints.disrtibute(solution) in the solve function of LocalProblem, there is no more coupling. Disabling it in Nonlocal and re-enabling it in LocalProblem. Adding norm output to determine if I have to send up or down.

The lower and upper interface norms give weird results. 

| Process Number | Lower interface norm | Upper interface norm |
| -------- | --------- | --------- |
| 1 | 1.40273 | 1.51729 |
| 2 | 2.02703 | 1.64984 |
| 3 | 8.54735 | 1.83122 |
| 4 | 19.2088 | 2.35377 |

Now the solutions look good. The solution is still the complex conjugate of the actual solution but that doesn't matter for now. The next step is to implement the backward sweep as a copy of the forward one. It is important to ensure, that the inverse only is applied once to the last block in the sweeping direction.

As a first step I copied the contents of apply_sweep from the old implementation to a seperate document.

# Tuesday, 27th of july

The computation still does not converge. As a next step I will try running GMRES with sweeping in only one direction. I deactivated the downward sweep. 

# Thursday, 29th of july

Very sick yesterday (second shot of the vaccination), therefore I made no progress. Currently, the solution generated by the apply_sweep() function is completely distorted.

I fixed some issues and now have a nice wave in the interior. However, the signal is not coupled in correctly and therefore the solution only has an amplitude of 1.0e-6. The change I made was to remove the distribution of the constraints in the function that writes the result of apply_sweep() into its x_out argument. Doing the same in the input parsing function u_from_x_in(). That did not solve the problem. Tomorrow I will first force the right hand side and re-evaluate if the coupling term is propagated correctly.

# Friday, 30th of july

I have re-extended the Geometry to a larger computational domain. This will show more wavelengths of the solution in the output. Currently, the mesh has vastly different h in the different coordinate directions, which I wanted to fix that way.

Interestingly, the convergence looks completely diffrent in this case.

Remark: I will soon need to reactivate the adjoint-based optimization code and the different strategies. It is important that I checkout the latest state of those code-segments in the repo to evaluate the ammount of work required to do so. It should be minimal, but it never hurts to check. Also, an implementation of BFGS is required and should, again, be simple.

As an experiment, I have removed all distribute and set_zero calls. I will test how this works. No visible change. I will now try to see if the jump coupling works since it doesnt rely on Dirichlet data.

# Saturday, 31th of july

I changed some parameters in the input file to see if prescribe zero was causing the problem. Went back to Dirichlet, because jump didnt fix the problem either. Convergence looks the same as before. I tested one GMRES step vs 10 and the amplitude of the signal was the same. It is therefore not a degenerative issue that gets worse every step (The solution doesnt tend towards 0, it solves an incorrect problem).

Next, I removed the constraints.set_zero() and constraints.solve() calls in the solve() function of the local problem because the distributed information my effect the solver. That reduced the starting norm by a factor of nearly 3. Evaling the outputs...

I found a MASSIVE bug. In the LocalProblem::solve() function I called constraints.set_zero(rhs) instead of constraints.set_zero(solution). The bug is fixed but the error persists. Will have to evaluate local solutions manually tomorrow.

# Sunday, 1st of august

The document for the extension of my contract is now finished. It also contains a graph of what the projects structure looks like. The texts and the graph can be used in my dissertation, where I also want introductury sections that explain the work in simple terms.

I am now checking to see what the solver produces, if the constraints get distributed in the set_x_out_from_u function (the values should be correct before any functions get called otherwise the internal solver doesn't see the constraints I think). Also added a set_zero and distribute call to the vmult function, using the local constraint object.

# Tuesday, 3rd of august

Added a function that only solves the local problem for the global rhs in the solve part. I also removed the member u from the NonlocalProblem type. As a consequence, the functions setSolutionFromVector, setChildSolutionComponentsFromU and setChildRhsComponentsFromU no longer make sense. Will delete them after this run(done). Another problem is, that the logs dont work properly at the moment. The files other than main0.log remain empty. This should be an easy fix.

Once the solver works or I run out of ideas to fix it, I should also fix the timer output. The current timer output seems incorrect.

No difference using right vs left preconditioning. It makes sense to use right for now because it means that convergence is measured in the norm of the original space and not in the preconditioned one. I should introduce a variable later to choose this in the parameters.

The local solver still generates the right amplitude so the error must be in the sweeping implementation.

Found another bug. The sweeping implementation reused trace2 instead of using trace1 and trace2. As a consequence, the sweeping solver was corrupted. This might solve the issue. Running test.

I cannot seem to find the exact source of the error. I have disabled some of the influences of set_zero and distribute and don't currently have a real overview of which calls to transfer a surface are really necessary...

OK, so, status quo: In the current state, the amplitude is incorrect. It is safe to assume, that the local solver works. It generates a stable amplitude.

# Wednesday, 4th of august

I simplified the implementation of the apply_sweep function. There was also another instance of the set_zero / distribute calls in the vmult function, that I have now removed. testing. In the current implementation, the issue is, that there is again a discontinuity at the interfaces with decrease from both sides. I suspect that the distribute() call in the local solver sets the lower boundary zero. Then, after the weep, the distribute call after solve puts a zero for the lower process aswell because of the constraint declaring the 2 dofs as equivalent.

I am now using the part before the sweeping preconditioner to build up a forward sweep and check amplitudes to see, where the error arises. 

# Thursday, 5th of august

In the attempted direct solve, I was missing a sign change. Fixed and restarted. In the resulting solution, there were missing interface values, which is now adjusted for run82. Also, the files have been renamed to Precalc and all 4 processes now contribute. The solution of the manual scheme looks great now. I copied the trace copy to the sweeping function.

I am now reenabling the distribution of constraints onto the input vector for apply sweep. Currently, the amplitude is 0.5 and that might be because there is only a sweep in one direction so there might be no backward coupling ... Will check tomorrow.

# Sunday, 8th of august

I changed all occurences of distribute_local_to_global to use true as the last argument. 

# Monday, 9th of august

I will now attempt to vary the way I call vmult to make sure the right action happens. If this doesnt work soon I will have to make the sweep variable and run all possible combinations. I massively reduced the geometry sice (50% each direction in cell count) to increase the speed of computation for faster debugging. As a consequence, the amplitude has increased massively. Wondering if there is a numerical error here ...I decreased it again and the amplitude reduced again. Ran four tests for z layers 7,8,9 and 10. Results were the same for all except the first (7) with an amplitude of 2.2e-2.
Now doing the same for geometry size. Here the amplitude seems to vary very strongly. 2->2.1 doubled the amplitude.

|z length | amplitude (x 0.1)|
|-|---|
|2|	0,95|
|2,1 |	1,1|
|2,2 |	1,2|
|2,3 |	1,5|
|2,4 |	1,9|
|2,5 |	2,1|
|2,6 |	2,2|
|2,7 |	2,5|
|2,8 |	2,7|
|2,9 |	3,1|
|3|	3,4|
|3,1|	3,5|
|3,2|	3,6|
|3,3|	3,7|
|3,4|	3,7|
|3,5|	4|
|3,75|	3,7|
|4|	3,8|
|4,25|	3,5|
|4,5|	3,9|

# Tuesday, 10th of august

The best I could do yesterday was 0.4 as the amplitude dampening factor. Plots of the values listed above imply no interference behaviour. The solution is continuous but the input signal did not couple correctly. I also ran tests with fewer processes and got the same result so the issue does not seem to depend on the number of processes, leaving me to believe that sweeping actually works.

# Wednesday, 11th of august

So over the last few days I ran nearly 100 tests. 
The amplitude *does* depend on:
- the total length of the computational domain
It *does not* depend on: 
- properties of the PML
- the length of idividual blocks (2x2 or 4x1 mu m block length deliver the same result)
- The number of cell layers per subdomain (above a certain minimum)
The outcome is that the amplitude of the sweeping solution depends on the total length of the domain. 

I changed the trace_to_field function to also set the dofs in the boundary method associated with the surface passed in as an argument.

# Thursday, 12th of august

There is new code in the function that sets vector components in the child object. That code writes the field values into the dofs enumerated by the additional surface.

Now the behaviour has changed to account for subdomain sizes again. 

I think I found the solution. I have to split Dirichlet and non-Dirichlet constraints.

# Friday, 13th of august

Here we go again. I have come to the conclusion, that I need to split constraints into constraints *with Dirichlet data* and *without*. That has been implemented. Whenever assembling data outside of the preconditioner, I use the constraints with the Dirichlet-data, but inside the preconditioner, I have to use the constraints without. I think.

I have added a datatype for Dirichlet surfaces, that does all one would expect of it. (One more topic: PML domains still bend towards Dirichlet surfaces, this could be changed.) The problem persists equal to the way it did before.

# Saturday, 14th of august

Currently the amplitude sits at 0.2 for the config I run. The dirichlet surface works well (tested with the direct solver) and the solution is continuous without any calls to distribute.

# Monday, 15th of august

I finally think I know the base of the error: When computing the downward sweep, lets assume The solution on the two neighboring domains is already correct. I compute the rhs to be sent down. The value I send down would then be exactly the rhs for the correct solution. On the lower process, however, that would lead to a local solution that is close to the one that process already has computed. As a consequence, computing the solution for that rhs and substracting it removes a lot of legitimate signal from the solution. Instead, when computing what to send down, I should use the difference between what is the old solution and the new one.

# Tuesday, 16th of august

I tried some details about deactivating the Dirichlet-zero-values on the upper interface by commenting out lines in the EmptySurface class but the results were not helping the problem. As a consequence, I rolled them back. I also noticed that the IO operations are a lot faster on the office machine compaired to my home PC. It is very fast in general. I guess the WSL2 layer is slowing down the IO part too much.

# Wednesday, 17th of august

I split the vmult function into two. Vmult_up and vmult_down. Because the matrix block product cannot be executed in a clear block sense here, I have to ensure that any entries outside the range of the input and output vectors that are concernned here, are zero before and after the product. No change.

# Thursday, 18th of august

Yesterday I came up with further steps to debug. The current implementation is, to compute the solution (which will have an amplitude of around 0.16). The  I compute b - A * u * (1/1.6). The result is the solution term in Run 196. The dominant residual concentration is located at the interfaces between subdomains. It is also largest in the x-components.

# Saturday, 21th of august

I hav implemented the following: I now compute U_direct - U_sweeping without distribution of boundary values. In the solution I can clearly see, that there is a large concentration of the error along the interfaces. Since GMRES does not converge, a 0 field with localized errors would not be expected. So a light oscillation in the visual field is expected. However, the error spikes. The higher (+z) process is OK, the lower process appears to have the wrong boundary values. This implies that I should copy all the boundary values downwards at the end of the sweep. Implementing.

# Thursday, 26th of august

I made some mistakes and had to git-stash my code. Sadly this cost me some days content of this file. 

Currently I'm improving code-quality by checking for dead code and improving implementation quality wherever I can see good options for it. At the same time I'm running tests with the two possible priorities of dof ownership in the function that generates constraints for two sets of dofs (i.e. constrain the inner to the outer or the outer to the inner).

I appear to have found an inconsistency in the dof set computation that might account for the errors I see. There was an error in the implementation of the computation of interface dof sets. It is fixed now. Searching dowstream problems.

# Friday, 27th of august

I still don't know the exact error. The exact solution generates a term on the input interface which I don't understand because these dofs hava a constraint on them. However, there are terms there that are a difference between the iterative and direct solution. Other then that there is only a slow amplitude increase in the first step solution of the iterative solver versus the direct solution. The next function I want to implement is one that lists the norm of a vector by domain the dofs belong to. I then want to use that to list the domains in which the residual is concentrated.

I switched all calls to distribute_local_to_global() back to use false as the last argument. As a consequence, it should now only be a matter of setting the appropriate dofs 0. This should be easier to diagnose and fix in the long run.

In the current setup, there is only a error in the components related to the upper surface of a subdomain. Not at the input and not at lower sides.

# Sunday, 29th of august (early night)

It seems I am tracking down the proplems causing the erroneous residual. I will add a better wrapper for the output that wraps the functionality of writing multiple output files and a .pvtu file along with it. Then I can always output the computed solution and the residual without rerunning and recompiling.

That implementation is done. I now have to fix the error that states incompatible interface dof counts. This was just an error in the implementation of the test. The sets are compatible.

Adding a "_" to the filename so the .pvtus are listed at the top.

# Sunday, 29th of august (late night)

I am considering the following: To remove all double numbering of dofs and instead of implementing complex logic to handle the constraints, instead just name them once and handle the numbering as logic. The numbering would be done exclusively by me in my code so I would not have to bother with library functions with weird documentation.
This would entail:
- Reimplementing the dof_numbering. This would also mean to split the implementation for the inner domain since the dof_numbering would be different on the various levels.
- The boundary methods would now only need the dofs that aren't shared by the interior or other surfaces. 
- The matrix sparsity pattern would be more complex to handle because multiple processes would write to the same lines.

Advantages: 
- No more constraints.
- Lower number of dofs.
- Simpler (more by-the-word) implementation of the numerical scheme.
- Easier to test since all work happens in my code, not the library.

The commit before this work was "320ebbb..9d66a7a".

Within each datastructure, the dofs should be numbered locally, i.e. in a range [0,...,N-1] for N dofs. Additionally, each structure should have an IndexSet indices with indices.n_elements() = N-1.

# Monday, 30th of august

I have started the implementation. The main work should be done in the class FEDomain which will be a base type for all structures that have dofs. It will manage the computation of the index numbering after everything is prepared. Also, all the domains must now be natively able to compute their correct dof counts (owned and active).

To simplify the implementation, I will also go ahead and add an inner_domain to every level instead of having this object only once. The implementation is fast enough to not have to worry about performance losses.

# Tuesday, 31st of august

I'm slowly grinding through the code to change the dof ownership. This has effects in extremely many places.

# Wednesday, 1st of september

I have implemented a huge amount of the new stuff. I have made the decision that get_dof_association() and get_dof_association_by_boundary_id() will always return in LOKAL dof numbering ( i.e. [0, ..., NDOFS - 1]). I will then add a function to the FEDomain set_global_dof_indices_by_boundary which sets(std::vector<InterfaceDofData>) to set the global indices. 

I have decided to go this way because all the calls to get_dof_association will now only happen in the setup. Therefore I can also repurpose them to the function I need in the setup rather then rewriting them.

# Saturday, 4th of september

By now, the local level initializes but the code breaks in the non-local initialization. Refactored the Geometry Manager to align function names with content and to make the structure more clear. That structure works now. Fixing dof commmunication of neighbor surfaces. Initialization now finishes.

# Monday, 6th of september

Most of the core is implemented. I realized that there is an error where apparently the same number as surface dofs of the inner domain is not generated properly on the surfaces. This may either be the interior surface not being initialized properly or it is the outer surface.

The order should be as follows: 
- Determine dof counts everywhere (in parallel)
- Receive on all lower neighbor surfaces
- Initialize all local counts
- Send all upper neighbor surfaces

The first step can be done in parallel. The important point here is, that every surface and inner domain can determine if a dof is locally owned or not so dof counts can be figured out from the beginning.
Receiving from below is blocking so this would wait until the lower process(es) have determined all dof indices. Once all lower surfaces have received from their counterpart, all dofs on the level can be numbered as in the necessary information is available. Once that is done, the computed global indices of the upper interface can be sent upward on the neighbor interfaces.

The first step is handled by the call to initialize_dof_counts() and determine_non_owned_dofs() where the first sets the numbers of dofs and initializes the datastructure and the second decides which dof belongs where. Next I call receive on all the neighbor surfaces because they will do nothing if they are upper ones.

The next step is to distribute all local dofs. This is the function finish_initialization() where I pass in the first_own_index in old parlance and initialize all locally owned dofs. Then the dofs need to be distributed such that all the other FEDomains that see this one will receive those values.

In the very last step, all neighbor surfaces send up.

# Tuesday, 7th of september

The above logic will be implemented today. It is done. Now I reach vector initialization.

# Wednesday, 8th of september

The dof distribution now works. The next step is building the sparsity pattern, wich is complicated, because the lower dofs don't know about the pattern entries the higher processes make. To do this, I must serialize the non-local contribution and send it the the right process.
The Neighbor surface has all data for this and should handle it. Now reaching pc_apply ao all the initialization seems to work properly.

# Saturday, 11th of september

I have decided to implement the sweeping preconditioner as a collective operation. That means, there will be no copying to a local vector and the loop will have a different structure compared to before (I had this once experimentally). The difference is then, that for example the off-diagonal block matrix vector product will be executed on all processes - they simply set zero in their vector for the product. If I only set the components of process i and only read the components of the solution of subdomain j, then I have evaluated a off-diagonal product for the (i,j) block.

This has to be done, because vmult on the problem-level matrix is itself a collective call, so performing it for only one block would not work otherwise. If there ever were problems with this approach, I could redefine the matrix and vectors to be block-structures and then just perform a call on a block but for now, a collective implementation will suffice.

I will go in more detail tomorrow or on monday, when I can check what is available in the library concerning vector-operations.

# Monday, 13th of september

Today I will begin the implementation mentioned above. It will require restructuring the Hierarchical Problem and both its derivatives. I could also butcher the new features in quick and dirty, but this is an opportunity to do it right. Since there will be no more distinguising between local and distributed vectors, I will need less utility functions and the logic should be clearer after this rework. This is why I want to work on this more wholistically, because if I don't go that route, there will be a lot of old code, that isn't used anymore. I should also consider static code analysis to find unused code at this point.

I will start by updating the headers and removing everything that is no longer required. Then I will prototype the functions that will be required after the rework.

The functions that do most of the work will be collective and I will preface their name with "collective_".

In the apply_sweep function I need:
- off-diagonal product which is implemented as a global operation
- S^{-1}, where I copy all locally owned dofs downwards and set the others to 0. Then the lower level solver gets executed and the solution gets copied up.
- A vector substraction function, that updates the local entries of the vector.

I can use the dealii::LinearAlgebra::distributed::Vector class to perform data synchronization between processes. It provides read from ghost values, so I can pass the locally_active_dofs into the constructor as the values I want synced and then just call update_ghost_values() to retrieve the data.

I have now removed all the code I wont need anymore, which was a lot. The code is currently compilable. The apply_sweep function is empty. I reactivated the direct vs iterative switch in solve.

Updated the output code to use the direct solver flag again. Also, the output now works again I think. However, the Dirichlet data is not yet working. The code runs completely again but the solution of a direct solve is not correct.

# Tuesday, 14th of september

Ok, so there is obviously an error int he way that either the dof-numbering or the solution output is implemented. Current symptoms:
- The solution behaves weird. The amplitude for real and imag part are in phase with eachother and the amplitudes are both wrong (real too big, imag too small).
- There are discontinuities at every boundary interface.
Both of these errors are important and they might influence eachother.
I am really stuck at the moment so I will see if the problem is in the distribution of dofs. To figure this out, I will write output that will print the dof indices of every dof. Then I will run this with a mini example consisting of only one cell inner and one cell layer for PML.
I found an error in the distribution of dof numbers which is fixed now. This fixed the issue. The computation is now correct again.
On ot the itterative solver. I will need to implement all the operations as global operations, which will make interacting with datastructures a lot simpler because I will allways use global numbering. There will only be copies when communicating with the level below. Also: I have to decide if I want to use a replacement vector for the internal work or the actual VEC object I am supposed to put the solution into.

# Wednesday, 15th of september

Initial Sweeping implementation done. Currently, there is some indexing problem in S_inv's call to set_child_rhs which I will figure out next. The entire sweep application is implemented as a collective operation.

# Monday, 27th of september

Some updates in this file were lost. But the implementation is looking stable so far. The only issue is currently the amplitude in the solution. I will implement some more output about the vectors (norm by process etc.).
After replaceing the vector get and set operations, this part appeared to work and now, the amplitude is correct.

# Tuesday, 28th of september

The problem is solved now. The remaining issue is, that the boundary methods are not the same on all levels. The angeling for edge PML cells can cause 2 errors:
1. They have different values than a straigt edge dof on the higher level would have.
2. They are sorted differently than the straight ones.
For these two reasons I cannot just copy the values between levels. I would copy wrong values and put them into wrong locations.

As a result, I have to rethink how I deal with this case. This solution has to be evaluated for both PML and HSIE. My first instinct is to simply leave all edges on PML domains straight and all non-interface dofs locally owned in the boundary method. This is a suboptimal solution because there are no boundary conditions on the sideway interfaces of the pml domain which makes the problem underdetermined. I will try this solution for both HSIE and PML and see how the solutions turn out. If they are insuficient in convergence rate or non-physical solutions I can change that. 

For PML I could consider to keep PML-edges the same on all levels but angled on the highest level. This would mean that a solution on the highest level would be physically viable, because on that level the boundary methods are appropriate and the system is not underdetermined. Also, in that case, the right-hand side terms might fix the issue of missing boundary conditions because the off-diagonal products essentially imply a boundary condition on those sides.

To implement this solution I introduce the value is_isolated on BoundaryCondition type objects. If this is set to true, the edges are straight and the dofs are all locally owned (except the ones facing the interior). It will not couple to any other boundaries. This is the case if this surface is a NeighborSurface on higher sweeping levels.

I removed all identify_corner_cells calls because the functions and the generated content had no effect anymore. It does currently build.
In a build later on, the first step reduced the residual by 95%. After that, convergence became stale.

# Wednesday, 29th of September

Today, I will reactivate some other parts of the implementation:
- The HSIE implementation to see if sweeping works with it
- Timers (Their output is somehow broken at the moment)
- The hierarchical part
- The shape optimization implementation
- The adjoint shape gradient computation

With all those pieces back in play I should be able to see what my scheme is capable off and plan a migration back to HPC systems with a fully operational system.

Starting with the HSIE implementation. I have added an output of the last git commit as a marker I can use in the future to see, with wich version I generated results. Returning to fixing HSIE.

Fixed all compiler warnings in Code (test still contains some about definitions of undefined functions). Currently, the direct solution is 0 and the preconditioners struggles extremely when switching to HSIE. There seems to be some issue in assembling the matrix. The numbers of owned and non-owned dofs look fine.

I removed some old instances of inner_first_index that might have caused the issue. In the local solution, the wave is visible and has the right amplitude for the real part. The imaginary part doesn't. There seems to be reflection. I also fixed some misunderstandings in get_dof_association and get_dof_assiciation_by_boundary_id.

# Thursday, 30th of September

I worked all day on regression testing the zero solutions. The bug has been found and fixed. Current status: For PML, there is no full convergence. For HSIE, the direct solutions is distorted. Somehow the non-X components of the solution are too large and as a consequence, there are weird distortions in the solution. The real part also overshoots.

I have had some additional ideas: 
- The first layer of the PML should have no PML material. That would have the advantage of it not effecting the dofs directly at the surface. This might improve convergence behaviour because the surface dofs couple into that first layer and are therefore subject to a decrease in field amplitude. The implementation should be straight forward. (Done)
- I need a solid setup to run my code remotely on my KIT PC because of RAM constraints. (Done)
- I should add output for the number of inner dofs vs boundary method dofs for the non-local and local problem and output them. This can be done pretty easily. (Done)
- I should run tests for higher element orders. (Done)
- For HSIE I should write code to compute the norm of the non-nedelec dofs in the solution vector. This would help to see if there is something extremly weird going on.
- I should run tests for higher level sweeping.
- I can implement a search for the ideal geometry by also performing material optimization. The material tensors consist of two terms: epsilon_r for inside or outside the waveguide and the material tensor for the shape transformation. Currently, my adjoint based approach uses only optimizations in the parameters fed into the material tensor computation. I could also compute if a cell should be inside or outside the domain with an integral over the cell and \Delta epsilon = 1.

I have implemented the PML offset. The PML should now ignore the first cell layer. At a later point I should add a function checking that the PML actually consists of more than one layer.
Next I will implement output for the number of dofs by type. 
Running test with the new PML safe layer, I see that there is still no convergence. Will analyze later.
I also added the dof count outputs.

I ran the PML code for higher order. In total there are 400K dofs in the 10x10x10 system with boundary methods. Laptop suffers. Since I no longer rely on all the code to compute interface dof sets, it seems that the code is now more stable. Assembly for boundary methods in the distributed case (level 1) seemst to work nicely. Assembly runs completely for the higher level and takes about 10 Minutes. The run works but there is no convergence. It is extremely stale (30 steps to go from 28 to 23).

About the erroneous dofs on interfaces. The dofs that cause the residual are the upper interface dofs of a domain - mainly in the boundary methods.The further out you go, the bigger the error. I will have to check if these get assembled correctly (non-transformed etc.).
It is good to see, that the code runs for higher order elements but it is not yet stable. I can analyze the run results to see if there are obvious problems. Could not connect via TeamViewer. Will have to retry tomorrow.

# Friday, 1st of October

I removed some old functions. During that, I noticed that most likely I no longer need to sort the dofs. That was needed before I reforged the meshes. As a consequence, this logic might actually cause the issues I see at interfaces by reorienting dofs the wrong way. Testing.
I also evaluated the results of the higher order run. The results look pretty similar in their errors to the HSIE problems on lowest order. This leads me to believe it's a problem with face orientations as mentioned above. There seems to be no issue without the sorting. Will now reattempt the higher order computations and HSIE. Lowest order and HSIE with sweeping gives a 0 solution. The solution is not exactly zero but of order e-5.
On the other hand: Order one computation for PML works now, including sweeping which is big progress. It even seems to deal with the interface dof error better than the lower order case does.

So currently there are either one or two problems:
1. For PML, there is no convergence, because thereis residual at the interfaces.
2. For HSIE, the solutions are completely wrong: because there appears to be incorrect coupling which introduces weird back scattering on cell layer level.

This could also be caused by the same issue.
To start again - I will check if the domain shape of the local problems is appropriate.  The shape of the local problems is correct. The isolated PML does not seem to cause trouble. However, it might be the case, that solving the local problem is the error. The "lower" dofs are not locally owned. In the upward sweep they are therefor zeroed after the off-diagonal product. This could cause problems down the road. I implemented a first solution to this based on MPI_Allgather in set_child_rhs_from_vector.

# Saturday, 2nd of October

The implementation to carry over the non-owned dofs seems to work but it didnt fix the issue. However, it may be, that I have another problem: If there is a neighbor surface, it only generates constraints for it's own dofs (i.e. the locally active dofs). This should however also happen for the boundary dofs of adjacent boundary methods. Implementing. This was already implemented, so that was not the issue.

I just noticed, that the first step is the one that works but the first step is also the one, where the downward sweep has no effect (starts with 0). So I will see what happens if the downward sweep is disabled completely. If that makes no difference, there is an issue in the implementation of the downward sweep. If not, I will have to keep searching. It would fit because then, the communication in the downward direction would only occur through GMRES which would explain the slow but existing convergence rate.

I have also undone the changes in the communication pattern because they were irrelevant.

# Monday, 4th of October

So far I cant tell yet what the issue really is. I will now implement a function that lists the dof ranges. Maybe I should write a MPI test suite to run tests on larger example cases. This would be a good issue for today. I will print on every process for the Boundary ID 1 what the highest and lowest dof on Interface 1-4 and 1-5 is.

# Wednesday, 6th of October

I spent all of yesterday, trying to find out what the source of the errors for PML might be. I was not successful. The runs for higher level sweeping currently don't work because of issues in boundary method dof ownership. I have reimplemented that part and hope, that it is now more reliable. There is now a static function to determine the ownership and the results get persisted in the BoundaryMethod objects. There is also an array with adjacent boundaries to allow for the more simplistic implementation with
```
for(auto surf: adjacent_boundary_ids) {
    ...
}
```
instead of the older 
```
for(unsigned int surf = 0; surf < 6; surf++) {
    if(surf != b_id && !are_opposing(b_id, surf) {
        ...
    }
}
```

The code now starts the run but gets stuck in the preconditioner. Evaluating why. The problem is, that the ranks are unique which makes sense in every application so far. However, in the level 2 sweep, multiple processes have to perform a S_inv at the same time. This means, they have to react as if they had the same rank. At the same time, there are some operations, where the unique rank counts. I therefore need a new value that is the index_in_sweeping_direction. Then all the processes with the same value here perform an action at the same time. Quickly checking if the code still works for level 1. Then I'm done for the day. The new implementation of boundary dof index ownership seems much more stable and comprehensive. The work item for tomorrow is obvious. I also copied the seg-fault issue solution to the HSIE implementation to make it work again. Additional commit.

I forced the GMRES Solver from PETSC to never do any restarts since they destroy the schemes convergence. I also noticed that the residual depends heavily on the parameters of the PML, leading me to believe that maybe the problem is not in the implementation but simply in the parameters of the PML.

# Thursday,  7th of October

Today, the goals are (in order):
- Optimize the outputs of timers and convergence history. (Done)
- Optimize the PML parameters for fast convergence. (Done)
- Running the first, higher-level sweeping preconditioner. Ideally also level 3.

Once these blocks are done, I will focus on repairing whatever the current issue in the HSIE implementation is.
There is now a new output object, that can generate a gnuplot script and execute it to generate a convergence history. That history is stored in the solution directory and contains all runs of all non-local levels. The script works and I worked on some details of the output.

# Friday, 8th of October

The convergence study shows, that while the PML parameters have an impact on the convergence after the first step, they don't dominate it to an extent, where the expected convergence is attainable by only using the proper PML parameters. I could use this as an example for how much PML has to be tuned, but other than that, the important take-away is, that this isn't the error I was searching for.

I will now switch to FEDomain implementation of local norm output, to be able to list vector norms by affected domain. This will help me see what is going wrong in the downward sweep. Implementation done. Checking. I see that the norms are largest for high process index and small for small process index. 

I reimplemented all of apply_sweep. The result is exactly the same. It is now split into a downward sweep, an upward sweep and an application of the local inverse (the three-step variant rather than the 2 step variant).

# Saturday, 8th of October

I don't really know what the issue is anymore. The sweeps are implemented as they are described in the paper. What strikes me as odd is that the norm of the vectors increases during the sweep, there seems to be some kind of disparity in the process that makes the communication of the off-diagonal product result a source of error which it apparently is. However - I have no clue where that happens.
Also: The problem in the HSIE implementation persists, leading me to believe there is still something wrong in that implementation, too. I have not solved the higher level sweeping issue about the rank in the sweep either. Again as a list:
- Find the bug in the sweeping implementation
- Find the bug in the HSIE implementation
- Implement the computation of the rank-in-sweep and run the code with it.

Today, I will focus on the HSIE issue. I made some minor adjustments and removed some old code.

# Wednesday, 12th of October

I'm traveling home from Canada today, writing this on the plane. I have had an idea to figure out a propable cause for the current issue:
By switching the way the inhomogenous constraints are made, I find a dof, for which I know the exact value. The value with the max inhomogenous value can be used to reconstruct the actual norm of the vector by multiplying the vector with a constant factor.

What I thought might be the issue is also not the problem. The only thing that remains are the open side-surfaces of boundary conditions. I will have to come up with a PML implementation that fixes that problem by constructing a more complex PML domain and solving the correct problem. If that works, the same is the issue for Hardy.

Maybe I can merge the existing meshes, extract a boundary mesh and then extrude that mesh n_layers times. I am however not sure if merging meshes is a thing in dealii. Will have to check with internet. This would simplify the process. ALso, finding the boundary dofs will not be as simple because defining the boundary cells isn`t simple, but it is all possible and easier for PML than for HSIE.

This should be possible to do on thursday morning. I will use todays afternoon to research possibilities and prepare the implementation.

There is another point worth mentioning: I have decided to include the Doxygen Docu in my dissertation. I will need a way to be able to reference classes and segments of code from the tex document. That would enable me to give the user a more complete guide to what I've done and how it works. This, however, requires a setup, that generates the Doxygen in a Latex-conform way that can then be included and referenced. I will also have to research, how to generate doxygen output that can be embedded actively in Latex Documents.

The initial implementation of the PML will only be done for the surface pair 4-5 and lowest order sweeping. If that solves the problem, I know what I am dealing with and can invest the time to finish a more general implementation. But first I should truely check if what I think is really the issue. It would make sense but is not guaranteed.

# Thursday, 13th of October

The first relevant step is, that there will be no more "isolated" boundary methods. Instead, they will couple with the other surfaces again and consist of the original boundary method core plus connecting domains. During the setup of the boundary method this information exists so I can pass it in and use it to build the correct mesh.
The functionality of merging triangulations does not currently exist. As a consequence I need to build it myself. It will consist of the following steps:

- Start as usual with a surface mesh. 
- Build additional meshes for the surfaces facing the other boundary methods. All required information about this is available at the time, the constructor gets called.
- Merge the surface triangulations manually. That means:
    - Get lists of nodes and edges for all the triangulations.
    - Identify the nodes and edges that are the same in all the datastructures.
    - Build a total collection of nodes and edges.
    - Build a new initialization list for the 2D mesh.
    - Extrude the mesh.
- Run the reforge mesh functionality.
- Identify the corner areas. This is now more of a search than a geometric operation because the interfaces are all on the same surface of the mesh.

Test coverage is somewhat important here, because these pieces are error-prone.

This implementation implies, that edge angeling is now a global property. Somehow one could also do without for PML. I think this would make sense because it gets rid of weird cell shapes and sizes and tends towards more homogenous cell shapes and sizex. This might be difficult for HSIE. Have to check.

For one level sweeping this is simple. The surfaces 0-4 (minus and plus x and y surfaces) have straight edges towards the surfaces 4 and 5.

An alternative would be to only have one boundary method for PML, one object that does all the work. Then I would simply have to create a mesh that, combined with the interior domain, triangulates the entire computational domain. This would require a lot of highly specialized code and should not be done, I guess.

It makes sense to introduce the following hierarchy:
- 0 and 1 boundaries extend completely orthogonal.
- 2 and 3 boundaries additionally contain connecting blocks with the 0 and 1 boundaries.
- 4 and 5 boundaries additionally contain connecting blocks towards all other boundaries.

The computation of the PML Tensor is also a bit more complicated now. I have to consider the cells in whom the PMLs overlap. See PML paper for details. I can handle this somewhat efficiently by giving material or subdomain ids to the cells that help identify what is the case.

# Monday 18th of October

Today, I implement the boundary condition overlap for PML. I found out, that there is a mesh merging function for coarse grids and will now attempt to use it.

I have written a first implementation, it aborts pretty early on. The error is not in prepare_mesh.

# Monday 25th of October

During last week I have tried to fix the mesh implementation for the new PML domain shape. There were some problems along the way and some counts still don't match in all cases.

# Wednesday 3rd of November

Therre has been immense progress. PML now works for sweeping. The next steps will follow two paths:
1. Make all the other parts work with the Code again. That includes transformation optics code, optimization code, etc. Maybe even implement the hybrid optimization code.
2. Get HSIE to work, too. This would simply offer a parallel line of tests but isn't required short term and not even long term if it is difficult to acchieve.

In my work for now, I will focus on getting part 1 to work again. That first means to integrate all the material tensor computations again.

I need a global SpaceTransformation Object. This shoule be able to switch between normal and adjoint state, perform the optimization of shape and it should also contain all the shape information. I will wrap this into 

# Thursday 4th of November

I have decided to fix the HSP first. This will enable me to run larger computations and is in general a new feature that will form a core part of my dissertation once functional. It is also not far from being completed. The issues here are more along the lines of bugfixing rather than implementing new functionality.

To reiterate, if the sweeping level is 2, that means:
- On level 0 I use a direct solver / LocalProblem.
- On level 1 I use sweeping in the y-direction / NonLocalProblem. In the minimal example, procs [0,1] and [2,3] form working pairs. Sweeping interfaces are 2 (-y) and 3(+y).
- On level 2 I use sweeping in the z-direction / NonLocalProblem. All processes are involved on the highest level. Sweeping interfaces are 4 (-z) and 5 (+z).

I reimplemented the SurfaceType computation in the Geometry Manager because it was asigning incorrect types for higher sweeping.

The issue I was running into was, that in some of the input files, the sweeping level was set to 1 even though there was sweeping required in the z and y direction. That lead to an issue. I should test against this case or simply compute the required sweeping level.

The current block occurs in reinit_vector which appears to have an indexing issue. I found an error that caused the initialization twice multiple times. This should lead to a significant speedup and wasn't really visible before. Currently, only process 0 arrives at the call to reinit_rhs().

Some hours later: The error was pretty instructional. I was using MPI_Send and MPI_Recv without tags which apparently led to a situation for higher sweeps for the communication of sparsity pattern where multiple messages with the same tag overwrote eachother. As a consequence, some processes got stuck unable to determine the fact that their message had been handled. I replaced 0 as a tag with the sending process' rank. Then the messages are unique and the communication works as expected.

Assembly works completely but the solver gets stuck. Problem for future-me.