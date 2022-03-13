#pragma once

#include <sys/stat.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "../Core/Types.h"
#include "../Solutions/ExactSolution.h"
#include "../GlobalObjects/ModeManager.h"
#include "../Helpers/ParameterReader.h"
#include "../Helpers/Parameters.h"
#include "../Helpers/staticfunctions.h"
#include "./Sector.h"
#include "../MeshGenerators/SquareMeshGenerator.h"
#include "../Core/Enums.h"
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>
#include "../GlobalObjects/GlobalObjects.h"
#include "./FEDomain.h"

/**
 * \class InnerDomain
 * \brief This class encapsulates all important mechanism for solving a FEM
 * problem. In earlier versions this also included space transformation and
 * computation of materials. Now it only includes FEM essentials and solving the
 * system matrix.
 *
 * Upon initialization it requires structural information about the waveguide
 * that will be simulated. The object then continues to initialize the
 * FEM-framework. After allocating space for all objects, the
 * assembly-process of the system-matrix begins. Following this step, the
 * user-selected preconditioner and solver are used to solve the system and
 * generate outputs. This class is the core piece of the implementation.
 *
 * \author Pascal Kraft
 * \date 03.07.2016
 */
class InnerDomain: public FEDomain {
 public:
  SquareMeshGenerator mesh_generator;
  dealii::FE_NedelecSZ<3> fe;
  dealii::Triangulation<3> triangulation;
  DofHandler3D dof_handler;
  dealii::SparsityPattern sp;
  dealii::DataOut<3> data_out;
  bool exact_solution_is_initialized;
  NumericVectorLocal exact_solution_interpolated;
  unsigned int level;

  InnerDomain(unsigned int level);
  ~InnerDomain();
  void load_exact_solution();
  void evaluate();
  void store();
  void make_grid();
  void setup_system();
  void assemble_system(Constraints *constraints, dealii::PETScWrappers::MPI::SparseMatrix *matrix, NumericVectorDistributed *rhs);
  void Compute_Dof_Numbers();
  void solution_evaluation(Position position, double *solution) const;
  void adjoint_solution_evaluation(Position position, double *solution) const;
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id(BoundaryId b_id);
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_pattern, Constraints *constraints);
  void write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs);
  std::string output_results(std::string in_filename, NumericVectorLocal in_solution, bool apply_space_transformation);
  void fill_rhs_vector(NumericVectorDistributed in_vec, unsigned int level);
  DofCount compute_n_locally_owned_dofs() override;
  DofCount compute_n_locally_active_dofs() override;
  void determine_non_owned_dofs() override;
  ComplexNumber compute_signal_strength(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);
  ComplexNumber compute_mode_strength();
  FEErrorStruct compute_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);
  std::vector<std::vector<ComplexNumber>> evaluate_at_positions(std::vector<Position> in_positions, NumericVectorLocal in_solution);
  std::vector<FEAdjointEvaluation> compute_local_shape_gradient_data(NumericVectorLocal & in_solution);
  Tensor<1,3,ComplexNumber> evaluate_J_at(Position);
};
