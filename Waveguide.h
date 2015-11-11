/**
 * Die Klasse Waveguide
 * Diese Klasse umfasst das gros der Funktionalität. Sie besteht aus dem üblichen Deal.II-Ablauf: make_grid(), setup_system(), assemble_system(), solve() und output_results().
 *
 * Funktion: make grid():
 * Diese Methode berechnet die Diskretisierung und baut das Mesh. Dies ist auch die einzige Stelle im Code, an dem noch Parameter zur neukompilierung vorliegen. In dieser Methode kann man an den kommentierten Stellen einstellen, wie genau verfeinert wird, also wie oft welche Strategie genutzt wird. Es spricht nichts gegen eine Auslagerung in das Parameter-file nur änderte sich dieser Teil häufig in Hinsicht auf seine Struktur und deshalb wurde davon bisher abgesehen.
 *
 * Funktion: setup_system():
 * Diese Methode initialisiert die wichtigen Teile des Systems wie den dof_handler, das finite Element und die Matrizen. Außerdem werden hier die Randwerte berechnet.
 *
 * Funktion: assemble_system():
 * In make_grid() wurde das Mesh in Blöcke eingeteilt. Es gibt einen Parameter, der angibt, wieviele Threads genutzt werden sollen. Das Mesh wird dann in doppelt so viele Blöcke zerlegt und diese durchnummeriert. Es werden dann in einem Durchlauf parallel alle geraden Blöcke assembliert und dann alle ungeraden. Da die Blöcke in einem Schritt durch einen kompletten Block voneinander getrennt sind, kann es beim parallelen Schreiben in die System-Matrix trotz massiver Parallelisierung, nicht zu Konflikten kommen.
 *
 * Funktion: solve():
 * In dieser Methode werden die Löser mit Parametern versehen. Außerdem erfolgt hier die Vorkonditionierung. An diesem Punkt erfolgte wenig Anpassung mit Außnahme des slot-connectors für das Auslesen von Eigenwerten und Konfitions-Schätzungen.
 *
 * Funktion: output_results():
 * Diese Methode nimmt die Lösungen und gibt sie in verschiedenen Formaten aus. Es wird ein Ergebnis-Objekt erstellt, an das dann Vektoren mit Namen angehängt werden können. Für diese Objekt existieren mehrere Überladungen für verschiedene Ausgabe-Ziele wie beispielsweise Paraview (vtk) und gnuplot.
 *
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef WaveguideFlag
#define WaveguideFlag

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
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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

static Parameters GlobalParams;

template <typename MatrixType, typename VectorType>
class Waveguide
{
	public:
		Waveguide (Parameters &, WaveguideStructure &);
		void 		run ();
		void 		rerun ();
		void 		assemble_part (unsigned int in_part);
		double 		evaluate_in();
		double 		evaluate_out();
		double 		evaluate_overall();
		void 		store();
		Tensor<2,3, std::complex<double>> get_Tensor(Point<3> &, bool, bool);



	private:
		void 	read_values ();
		void 	make_grid ();
		void 	setup_system ();
		void 	assemble_system ();
		void	estimate_solution();
		void 	solve ();
		void 	reset_changes();
		void 	output_results ();
		void 	print_eigenvalues(const std::vector<std::complex<double>> &);
		void	print_condition (double);
		bool	PML_in_X(Point<3> &);
		bool	PML_in_Y(Point<3> &);
		bool	PML_in_Z(Point<3> &);
		double 	PML_X_Distance(Point<3> &);
		double 	PML_Y_Distance(Point<3> &);
		double 	PML_Z_Distance(Point<3> &);
		void	MakeBoundaryConditions ();
		double  RHS_value(const Point<3> &, const unsigned int component);
		Tensor<2,3, std::complex<double>> Transpose_Tensor(Tensor<2,3, std::complex<double>> );
		Tensor<1,3, std::complex<double>> Transpose_Vector(Tensor<1,3, std::complex<double>> );
		void	init_loggers();
		void 	timerupdate();
		dealii::Vector<double> differences;

		std::string			solutionpath;

		Triangulation<3>	triangulation, triangulation_real;
		FESystem<3>			fe;
		DoFHandler<3>	dof_handler, dof_handler_real;
		VectorType		solution;
		ConstraintMatrix 	cm;

		SparsityPattern		sparsity_pattern;
		MatrixType			system_matrix;
		Parameters			&prm;
		ConstraintMatrix 	boundary_value_constraints_imaginary;
		ConstraintMatrix 	boundary_value_constraints_real;

		int 				assembly_progress;
		VectorType			storage;
		bool				is_stored;
		VectorType			system_rhs;
		FileLoggerData 		log_data;
		FileLogger 			log_constraints, log_assemble, log_precondition, log_total, log_solver;
		WaveguideStructure 	&structure;
		int 				run_number;
		int					condition_file_counter, eigenvalue_file_counter;
		std::ofstream		eigenvalue_file, condition_file, result_file;

};


#endif
