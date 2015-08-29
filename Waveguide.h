#ifndef WaveguideFlag
#define WaveguideFlag

/**
 * This Code is the core file of the Waveguide-Problem Solver. It is the main part of my Masters Thesis implementation.
 * The Goal of this project is, to solve an optimization problem concerning the perfect shape for minimal signal-loss in an S-shaped waveguide.
 * The solver is implemented using the code-framework DealII (see <a href="http://dealii.org/">here</a> for more information). Mathematically the main idea is, to focus on the shape-representation problem. One  of the biggest problem in any solution of this problem, is to represent the used shapes optimally. In this case we will always use the same discretization because we use transformation optics to transform the real physical problem on a s-shaped waveguide with simple material-parameters, to a mathematical representation, consisting of a cylindrical quasi-waveguide made up of an inhomogenous material.
 *
 * @author Pascal Kraft
 * @date 2015
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_matrix_base.h>
#include <deal.II/lac/block_vector_base.h>
/*
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
*/
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <mpi/mpi.h>
#include <sstream>

#include "Parameters.h"
#include "ParameterReader.h"
#include "WaveguideStructure.h"
#include "FileLogger.h"
#include "FileLoggerData.h"

using namespace dealii;

template <typename MatrixType, typename VectorType>
class Waveguide
{
	public:
		Waveguide (Parameters &, WaveguideStructure &);
		void 		run ();
		void 		assemble_part (unsigned int in_part);
		double 		evaluate_in();
		double 		evaluate_out();
		double 		evaluate_overall();
		void 		store();

	private:
		void 	read_values ();
		void 	make_grid ();
		void 	setup_system ();
		void 	assemble_system ();
		void	estimate_solution();
		void 	solve ();
		void 	output_results () const;
		bool	PML_in_X(Point<3> &);
		bool	PML_in_Y(Point<3> &);
		bool	PML_in_Z(Point<3> &);
		double 	PML_X_Distance(Point<3> &);
		double 	PML_Y_Distance(Point<3> &);
		double 	PML_Z_Distance(Point<3> &);
		double  RHS_value(const Point<3> &, const unsigned int component);
		Tensor<2,3, std::complex<double>> get_Tensor(Point<3> &, bool, bool);
		Tensor<2,3, std::complex<double>> Transpose_Tensor(Tensor<2,3, std::complex<double>> );
		Tensor<1,3, std::complex<double>> Transpose_Vector(Tensor<1,3, std::complex<double>> );
		void	init_loggers();
		void 	timerupdate();
		std::string		solutionpath;

		Triangulation<3>	triangulation;
		FESystem<3>			fe;
		DoFHandler<3>		dof_handler;
		ConstraintMatrix 	cm;

		SparsityPattern		sparsity_pattern;
		MatrixType			system_matrix;
		Parameters		&prm;
		ConstraintMatrix boundary_value_constraints_imaginary;
		ConstraintMatrix boundary_value_constraints_real;

		int 			assembly_progress;

		VectorType	solution;
		VectorType	storage;
		bool		is_stored;
		VectorType	system_rhs;
		FileLoggerData log_data;
		FileLogger log_constraints, log_assemble, log_precondition, log_solver, log_total;
		WaveguideStructure structure;

};


#endif
