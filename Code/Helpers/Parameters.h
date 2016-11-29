
#ifndef ParametersFlag
#define ParametersFlag

/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters available globally.
 * \author: Pascal Kraft
 * \date: 28.11.2016
 */
struct Parameters {
	/**
	 * Output more details about the grid.
	 */
	bool			PRM_O_Grid;
	/**
	 * Print the number of dofs.
	 */
	bool			PRM_O_Dofs;
	/**
	 * Print the number of active cells.
	 */
	bool			PRM_O_ActiveCells;
	/**
	 * Generally write more details about the run to the console.
	 */
	bool 			PRM_O_VerboseOutput;
	/**
	 * Shape of the input-connector. Currently only circular inputs are possible.
	 */
	std::string		PRM_M_C_TypeIn;
	/**
	 * Shape of the output-connector. Currently only circular inputs are possible.
	 */
	std::string 	PRM_M_C_TypeOut;
	/**
	 * In case of a circular input connector, this variable holds its radius.
	 */
	double			PRM_M_C_RadiusIn;
	/**
	 * In case of a circular output connector, this variable holds its radius.
	 */
	double			PRM_M_C_RadiusOut;
	/**
	 * Length of the computational domain in the \f$x\f$-direction.
	 */
	double				PRM_M_R_XLength;
	/**
	 * Length of the computational domain in the \f$y\f$-direction.
	 */
	double 				PRM_M_R_YLength;
	/**
	 * Length of the computational domain in the \f$z\f$-direction.
	 */
	double				PRM_M_R_ZLength;
	/**
	 * Distance of the middlepoints of the input- and output-connector. The direction of the light propagation in the input-wavegudie is used as the \f$z\f$-direction. The shift of the two connectors orthogonally to the \f$z\f$-direction is used as the \f$y\f$-direction.
	 */
	double			PRM_M_W_Delta;
	/**
	 * This variable holds the material-property \f$\epsilon_r\f$ of the material inside the waveguide-vore, which is assumed to be a linear material. Keep in mind, that at this point, \f$\epsilon_r\f$ has to be a scalar value.
	 */
	double			PRM_M_W_EpsilonIn;
	/**
	 * This variable holds the material-property \f$\epsilon_r\f$ of the material outside the waveguide-core, which is assumed to be a linear material. Keep in mind, that at this point, \f$\epsilon_r\f$ has to be a scalar value.
	 */
	double			PRM_M_W_EpsilonOut;
	/**
	 * The wavelength of the light sent into the system via the input connector.
	 */
	double			PRM_M_W_Lambda;
	/**
	 * Type of boundary-conditions to be used. Currently, only a PML-method is implemented.
	 */
	std::string		PRM_M_BC_Type;
	/**
	 * Thickness of the PML-region on the input-side of the system. In most cases this will be zero since there are Dirichlet values known for this interface.
	 */
	double			PRM_M_BC_XYin;
	/**
	 * Thickness of the PML-region of the output-side of the system. In most cases this will be rather large, since it is supposed to absorb the complete signal without any reflections. The numerical error in this scheme reduces, if the PML-region is not too short.
	 */
	int			PRM_M_BC_XYout;
	/**
	 * Thickness of the PML-region along the waveguide. This parameter is quite flexible and should be tuned appropriately.
	 */
	double			PRM_M_BC_Mantle;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors real components for the PML in \f$x\f$-direction.
	 */
	double			PRM_M_BC_KappaXMax;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors real components for the PML in \f$y\f$-direction.
	 */
	double			PRM_M_BC_KappaYMax;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors real components for the PML in \f$z\f$-direction.
	 */
	double			PRM_M_BC_KappaZMax;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors imaginary components for the PML in \f$x\f$-direction.
	 */
	double			PRM_M_BC_SigmaXMax;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors imaginary components for the PML in \f$y\f$-direction.
	 */
	double			PRM_M_BC_SigmaYMax;
	/**
	 * This is one design-parameter of the PML determining the maximal value of the PMLs material tensors imaginary components for the PML in \f$z\f$-direction.
	 */
	double			PRM_M_BC_SigmaZMax;
	/**
	 * Another design-parameter of the PML, describing the degree of polynomial increase of the real and imaginary part of the material-tensor over the distance to the PML-regions boundary towards the interior.
	 */
	int				PRM_M_BC_M;
	/**
	 * This value stores the method of refinement. This is not currently used in the programm.
	 */
	std::string 	PRM_D_Refinement;
	/**
	 * Number of mesh-refinement steps done in the \f$xy\f$-plane.
	 */
	int 			PRM_D_XY;
	/**
	 * Number of refinement steps done in the \f$z\f$-direction.
	 */
	int				PRM_D_Z;
	/**
	 * Number of steps the solver is allowed to use before throwing an error due to non-convergence.
	 */
	int 			PRM_S_Steps;
	/**
	 * Number of steps GMRES should do before deleting its basis and doing a restart. Without very good preconditioning, this value has to be very large to achieve convergence.
	 */
	int				PRM_S_GMRESSteps;
	/**
	 * For Preconditioners based upon a block-scheme, this is the number of blocks they should use.
	 */
	int				PRM_S_PreconditionerBlockCount;
	/**
	 * The number of threads to be used to assemble the system matrix.
	 */
	int				PRM_A_Threads;
	/**
	 * Value of the residual for which to stop due to convergence.
	 */
	double			PRM_S_Precision;
	/**
	 * Solver to be used to solve the system-matrix.
	 */
	std::string 	PRM_S_Solver;
	/**
	 * Preconditioner to be used in the solution-process of the system-matrix.
	 */
	std::string		PRM_S_Preconditioner;
	/**
	 * Number of sectors to be used to model the waveguide.
	 */
	int 			PRM_M_W_Sectors;
	/**
	 * Constant containing the value of \f$\pi\f$.
	 */
	double 			PRM_C_PI;
	/**
	 * Constant containing the value of \f$\epsilon_0\f$.
	 */
	double			PRM_C_Eps0;
	/**
	 * Constant containing the value of \f$\mu_0\f$.
	 */
	double			PRM_C_Mu0;
	/**
	 * Constant containing the speed of light.
	 */
	double			PRM_C_c;
	/**
	 * Constant containing the value of \f$f_0\f$.
	 */
	double			PRM_C_f0;
	/**
	 * Constant containing the value of \f$\omega\f$.
	 */
	double			PRM_C_omega;
	/**
	 * Maximal amount of steps the optimization-algorithm should to before calling the current state the solution.
	 */
	int				PRM_Op_MaxCases;
	/**
	 * The initial step of the optimization-algorithm is calculated via
	 * \f[\boldsymbol{s} = \sigma \nabla Q(\boldsymbol{k}) \qquad k\in \mathbb{R}^{\operatorname{dofs}}\f]
	 * where dofs is the number of degrees of freedom of the shape and \f$\boldsymbol{k}\f$ stores the values of those degrees of freedom. \f$Q\f$ is the functional calculating the signal quality (this is a scalar value if only one mode is used). The exact value of the gradient cannot be calculated analytically but is approximated via difference-quotients upon doing small steps for all degrees of freedom and calculating the signal quality for the resulting system.
	 */
	double			PRM_Op_InitialStepWidth;
	/**
	 * The tilt towards the \f$z\f$-axis of the input-connector. Since this property being non-zero in combination with circular input-connectors makes the model of modes in the waveguide at least analytically obsolete, currently only 0 is allowed for this value. It is not impossible to model such systems, the issue is simply that the description of the signal based upon guided modes is no longer feasible.
	 */
	double			PRM_M_C_TiltIn;
	/**
	 * The tilt towards the \f$z\f$-axis of the output-connector. Since this property being non-zero in combination with circular input-connectors makes the model of modes in the waveguide at least analytically obsolete, currently only 0 is allowed for this value. It is not impossible to model such systems, the issue is simply that the description of the signal based upon guided modes is no longer feasible.
	 */
	double			PRM_M_C_TiltOut ;
	/**
	 * Constant containing the value of \f$k_0\f$.
	 */
	double			PRM_C_k0;
	/**
	 * Constant containing the value of the optical wavelength \f$\lambda\f$.
	 */
	double			PRM_C_lambda;

	/**
	 * Library to be used during calculation: One of either "DealII", "PETSc" or "Trilinos"
	 */
	std::string			PRM_S_Library;

	/**
	 * Number of MPI-processes to be used
	 */
	unsigned int 		PRM_S_MPITasks;

	/**
	 * This value states wether optimization should be attempted or if only the first shape should be calculated.
	 */
	bool				PRM_S_DoOptimization;

	/**
	 * This value sets the number of global refinements to be done. This also influeces the number of sectors since in the case \f$ 5 \cdot 2^{N_G} == N_S \cdot n \qquad n \in \mathbb{N} \f$ the sector-boundaries coincide with the cell-boundaries of the mesh.
	 */
	int 				PRM_R_Global;

	/**
	 * This value specifies how many semi-global refinement steps should be done.
	 */
	int					PRM_R_Semi;

	/**
	 * This parameter sets the amount of refinementsteps inside the waveguide to be performed.
	 */
	int					PRM_R_Internal;

	/**
	 * For the shifts and radii this is the upper boundary during optimization. It is relative to the larger of the boundary values of the relevant value. I.e. if radii are 3 and 2 and this is set to 1.1 then the value will not exceed 3.3 for any sector boundary during computation.
	 */
	double 				PRM_O_MaximumFactor;

	/**
	 * For the shifts and radii this is the lower boundary during optimization. It is relative to the smaller of the boundary values of the relevant value. I.e. if radii are 3 and 2 and this is set to 0.9 then the value will not fall below 1.8 for any sector boundary during computation.
	 */
	double 				PRM_O_MinimumFactor;

	/**
	 * This value stores the rank of the current process such that the function getting it doesnt have to be called several times.
	 */
	unsigned int	MPI_Rank;

	/**
	 * This is the MPI-communicator to be used globally.
	 */

	MPI_Comm	MPI_Communicator;

	unsigned int MPI_Size;


	/**unsigned int sub_block_lowest;

	unsigned int block_lowest;

	unsigned int block_highest;
	**/

	double z_evaluate;

	double z_min, z_max;

	bool evaluate_in;

	bool evaluate_out;
};

#endif
