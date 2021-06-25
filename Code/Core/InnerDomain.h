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
#include "../SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../Core/Enums.h"
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>
#include "../GlobalObjects/GlobalObjects.h"

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
class InnerDomain {
 public:
  SquareMeshGenerator mesh_generator;
  HomogenousTransformationRectangular space_transformation;
  dealii::FE_NedelecSZ<3> fe;
  dealii::Triangulation<3> triangulation;
  bool local_constraints_made;
  dealii::AffineConstraints<ComplexNumber> local_constraints;
  unsigned int n_dofs;
  DofHandler3D dof_handler;
  dealii::IndexSet fixed_dofs;
  dealii::IndexSet local_dof_indices;
  std::set<std::string> constrained_cells;
  std::set<unsigned int> outer_constrained_faces;
  dealii::SparseMatrix<ComplexNumber> local_matrix;
  dealii::SparsityPattern sp;
  bool is_local_matrix_prepared = false;

  InnerDomain();
  ~InnerDomain();
  double evaluate_for_z(double);
  void evaluate();
  void store();
  void make_grid();
  void setup_system();
  void assemble_system(unsigned int shift,
      dealii::AffineConstraints<ComplexNumber> *constraints,
      dealii::PETScWrappers::MPI::SparseMatrix *matrix,
      NumericVectorDistributed *rhs);
  void assemble_system(unsigned int shift,
      dealii::AffineConstraints<ComplexNumber> *constraints,
      dealii::PETScWrappers::SparseMatrix *matrix,
      NumericVectorDistributed *rhs);
  void Compute_Dof_Numbers();
  void solution_evaluation(Position position, double *solution) const;
  void adjoint_solution_evaluation(Position position, double *solution) const;
  void SortDofsDownstream();
  void make_constraints(dealii::AffineConstraints<ComplexNumber>*, unsigned int shift, IndexSet local_constraints_indices);
  void make_constraints();
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id(BoundaryId b_id);
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id_and_level(BoundaryId b_id, unsigned int level);
  std::vector<InterfaceDofData> get_surface_dof_vector_for_edge(BoundaryId first_bid, BoundaryId second_bid);
  std::vector<InterfaceDofData> get_surface_dof_vector_for_edge_and_level(BoundaryId first_bid, BoundaryId second_bid, unsigned int level);
  std::vector<SurfaceCellData> get_surface_cell_data_for_boundary_id_and_level(BoundaryId b_id, unsigned int level);
  std::vector<unsigned int> dofs_for_cell_around_point(Position &in_p);
  void make_sparsity_pattern(dealii::DynamicSparsityPattern *in_pattern, unsigned int shift, dealii::AffineConstraints<ComplexNumber> *constraints);
  void write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs);
  auto get_central_cells(double point_source_radius) -> std::set<std::string>;
  auto get_outer_constrained_faces() -> std::set<unsigned int>;
  std::vector<SurfaceCellData> get_edge_cell_data(BoundaryId first_b_id, BoundaryId second_b_id, unsigned int level);
  void output_results(std::string in_filename, NumericVectorLocal in_solution);
  void fill_rhs_vector(NumericVectorDistributed in_vec, unsigned int level);
  void prepare_inner_matrix();
  auto vmult(const NumericVectorLocal a) -> NumericVectorLocal;
};
