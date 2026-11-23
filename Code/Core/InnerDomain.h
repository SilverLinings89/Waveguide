#pragma once

/**
 * @file InnerDomain.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief Contains the implementation of the inner domain which handles the part of the computational domain that is locally owned.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <string>
#include <vector>
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
 */
class InnerDomain: public FEDomain {
 public:
  /** @brief Mesh generator object. It is only constructed; make_grid() builds the mesh directly. */
  SquareMeshGenerator mesh_generator;
  /** @brief The Nedelec element of the order set in the parameters (Nedelec_element_order). */
  dealii::FE_NedelecSZ<3> fe;
  /** @brief Mesh of the part of the inner domain owned by this process, in computational ("math") coordinates. */
  dealii::Triangulation<3> triangulation;
  /** @brief Dof handler on triangulation. Its numbering is the local numbering of this domain. */
  DofHandler3D dof_handler;
  /** @brief Sparsity pattern. Not used by the functions of this class. */
  dealii::SparsityPattern sp;
  /** @brief Output object used by output_results(). */
  dealii::DataOut<3> data_out;
  /** @brief True once load_exact_solution() has computed exact_solution_interpolated. */
  bool exact_solution_is_initialized;
  /** @brief L2-projection of the source field (GlobalParams.source_field) onto the finite element space, in local numbering. */
  NumericVectorLocal exact_solution_interpolated;
  /** @brief The sweeping level this object belongs to, i.e. the index into Geometry.levels. */
  unsigned int level;

  /**
   * @brief Creates the finite element and the (still empty) triangulation and dof handler.
   *
   * @param level The sweeping level this object belongs to.
   */
  InnerDomain(unsigned int level);
  /** @brief Empty destructor. */
  ~InnerDomain();
  /**
   * @brief In many places it can be useful to have an interpolated exact solution for the waveguide or Hertz case.
   * This function ensures the analytical solution is available and projects it onto the FE space to compute a solution vector.
   * 
   */
  void load_exact_solution();

  /**
   * @brief This function builds the triangulation for the inner domain part on this level that is locally owned.
   * 
   */
  void make_grid();

  /**
   * @brief Main part of the system matrix assembly loop. 
   * Writes all contributions of the local domain to the system matrix provided as a pointer.
   * 
   * @param constraints All constraints on degrees of freedom.
   * @param matrix The system matrix to be filled.
   * @param rhs The right-hand side vector to be used.
   */
  void assemble_system(Constraints *constraints, dealii::PETScWrappers::MPI::SparseMatrix *matrix, NumericVectorDistributed *rhs);

  /**
   * @brief Returns a vector of all dofs active on the given surface.
   * This cna be used to build the coupling of the interior with a boundary condition.
   * 
   * @param b_id The boundary one is interested in.
   * @return std::vector<InterfaceDofData> The vector of dofs on that surface.
   */
  std::vector<InterfaceDofData> get_surface_dof_vector_for_boundary_id(BoundaryId b_id);

  /**
   * @brief Marks all index pairs that are non-zero in the provided matrix using the given constraints.
   * See the dealii documentation for more details on how this is done and why.
   * 
   * @param in_pattern The pattern to fill.
   * @param constraints The constraints to consider.
   */
  void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_pattern, Constraints *constraints);

  /**
   * @brief Prints some diagnostic data to the console.
   * 
   * @param matrix The matrix whose l1 norm is printed.
   * @param rhs The vector whose l2 norm is printed.
   */
  void write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs);

  /**
   * @brief Generates an output file of the provided solution vector on the local domain.
   * 
   * @param in_filename The filename to be used for the output. This will be made unique by appending process ids.
   * @param in_solution The solution vector representing the solution on the described domain.
   * @param apply_space_transformation If set to true, the output domain will be transformed to the physical coordinates.
   * @return std::string The actual filename used after making it unique. This can be used to write the fileset files.
   */
  std::string output_results(std::string in_filename, NumericVectorLocal in_solution, bool apply_space_transformation);

  /**
   * @brief Fulfills FEDomain interface. See definition there.
   * 
   * @return DofCount 
   */
  DofCount compute_n_locally_owned_dofs() override;
  
  /**
   * @brief Fulfills FEDomain interface. See definition there.
   * 
   * @return DofCount 
   */
  DofCount compute_n_locally_active_dofs() override;

  /**
   * @brief Fulfills FEDomain interface. See definition there.
   * 
   */
  void determine_non_owned_dofs() override;

  /**
   * @brief Computes how strongly the fundamental mode is excited in the output waveguide in the field provided as the input.
   * 
   * @param in_solution The solution to check this for.
   * @return ComplexNumber The complex phase and amplitude of the fundamental mode in the solution.
   */
  ComplexNumber compute_signal_strength(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);

  /**
   * @brief Computes the norm of the input mode for scaling of the output signal.
   * 
   * @return ComplexNumber 
   */
  ComplexNumber compute_mode_strength();

  /**
   * @brief Computes the L2 and L_infty error of the provided solution against the source field (i.e. exact solution if applicable).
   * 
   * @param in_solution The FE solution we want to compute the errors for.
   * @return FEErrorStruct A struct containing L2 and L_infty members.
   */
  FEErrorStruct compute_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution);

  /**
   * @brief Evaluates the provided solution (represented by in_solution) at the given positions, i.e. computes the E-Field at a given locations.
   * 
   * @param in_positions The positions we want to know the solution at.
   * @param in_solution The solution vector from the finite element method.
   * @return std::vector<std::vector<ComplexNumber>> The vector of field evaluations.
   */
  std::vector<std::vector<ComplexNumber>> evaluate_at_positions(std::vector<Position> in_positions, NumericVectorLocal in_solution);

  /**
   * @brief Computes point data required to compute the shape gradient. 
   * To compute the shape gradient, we require at every quadrature point of the evaluation quadrature:
   * - The primal solution
   * - The curl of the primal solution
   * - The adjoint solution
   * - The curl of the adjoint solution
   * - The location that these values were computed at.
   *  This function computes all these values and stores them in an array. Every entry is the data for one quadrature point.
   * 
   * @param in_solution The solution vector from the finite element method applied to the primal problem.
   * @param in_adjoint The solution vector of the finite element method applied to the adjoint problem.
   * @return std::vector<FEAdjointEvaluation> Vector of datasets for a quadrature of the local domain with field evaluations and curls.
   */
  std::vector<FEAdjointEvaluation> compute_local_shape_gradient_data(NumericVectorLocal & in_solution, NumericVectorLocal & in_adjoint);

  /**
   * @brief Computes the forcing term J for a given position so we can use it to build a right-hand side / forcing term.
   * 
   * @param in_p The position to evaluate J at.
   * @return Tensor<1,3,ComplexNumber> The complex vector containing the three components of J at the given location.
   */
  Tensor<1,3,ComplexNumber> evaluate_J_at(Position in_p);

  /**
   * @brief Computes the value \f$\kappa\f$
   * 
   * This value is defined by 
   * \f[ 
   * \kappa = \int_{\Gamma_O}\overline{\boldsymbol{E}_0}\cdot \boldsymbol{E}_p \mathrm{d} A
   * \f]
   * 
   * 
   * @param in_solution The solution \f$\boldsymbol{E}_p\f$ in local numbering.
   * @return ComplexNumber The value of the integral for the part of the output surface in this domain.
   */
  ComplexNumber compute_kappa(NumericVectorLocal & in_solution);

  /**
   * @brief Subtracts the contributions of the faces on the output surface (upper end of the global z range) to the right-hand side of an adjoint problem based on compute_kappa().
   *
   * For every dof i with shape function \f$\phi_i\f$ on such a face the value \f$2\,\mathrm{Re}(\kappa \, \boldsymbol{E}_0 \cdot \overline{u_i} \phi_i)\f$, integrated over the face, is subtracted, where \f$\boldsymbol{E}_0\f$ is the source field. The entries of in_rhs are addressed with the local dof indices of this domain. See assemble_adjoint_rhs_for_signal_strength() for the version that uses global indices.
   *
   * @param in_solution The primal solution in local numbering.
   * @param in_rhs The right-hand side vector to modify.
   */
  void set_rhs_for_adjoint_problem(NumericVectorLocal & in_solution, NumericVectorDistributed * in_rhs);

  /**
   * @brief Assembles the right hand side of the adjoint problem for the signal strength functional.
   * The signal at the output interface is S(u) = int_out u . conj(m) dA with the mode m, which is a linear
   * functional S(u) = sum_i G_i u_i with G_i = int_out conj(m) . phi_i dA. The adjoint state lambda solves
   * A^T lambda = G. This function adds the local part of G to the vector.
   *
   * @param in_rhs The (distributed) right hand side.
   * @return double The local part of the normalization int_out |m|^2 dA.
   */
  double assemble_adjoint_rhs_for_signal_strength(NumericVectorDistributed * in_rhs);

  /**
   * @brief Writes the field on three planes through the domain as plain text, for quick visualization without
   * ParaView (the web frontend reads these files).
   * The planes are x = 0, y = 0 (both through the axis of the waveguide) and the plane z = const in the middle
   * of the global domain. For every cell of this process that is cut by a plane, the field at the cell center is
   * written: position in mathematical coordinates, position in physical coordinates, real and imaginary parts of
   * the three components and whether the point is inside the waveguide core. One file per plane and process.
   *
   * @param in_solution The solution in the local numbering of this domain.
   * @param in_name Name of the output, the files are called `<in_name>_<plane><process>.txt`.
   */
  void write_field_slices(const NumericVectorLocal & in_solution, const std::string & in_name);

  /**
   * @brief Writes the overlap of the solution with the input mode for every plane of cell faces z = const.
   *
   * For every plane the file `<name><process>.txt` contains the sums over the faces (values at the face centers) of
   * E * conj(m), |m|^2, |E|^2 in the core and |E|^2 in the whole cross-section, where m is the mode at the same
   * point. The sums of all processes for one z give the transmitted signal s(z) = sum E conj(m) / sum |m|^2.
   *
   * @param in_solution The solution in local numbering.
   * @param in_name Base name of the file.
   */
  void write_signal_along_z(const NumericVectorLocal & in_solution, const std::string & in_name);

  /**
   * @brief Computes for every free shape parameter p_j the local part of
   *   lambda^T (dA/dp_j u - db/dp_j),
   * i.e. the integrals of (dT^-1 curl u) . curl lambda - omega^2 eps (dT u) . lambda over the cells of this
   * domain, minus the derivative of the tapered source term. T is the tensor of the space transformation and its
   * derivative with respect to p_j is computed by central differences. There is no complex conjugation: the form
   * is bilinear.
   *
   * @param in_solution The primal solution u in local numbering.
   * @param in_adjoint The adjoint solution lambda in local numbering.
   * @return One value per free shape parameter.
   */
  std::vector<ComplexNumber> compute_shape_gradient_integrals(NumericVectorLocal & in_solution, NumericVectorLocal & in_adjoint);
};
