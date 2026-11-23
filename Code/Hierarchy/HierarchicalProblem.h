#pragma once
/**
 * @file HierarchicalProblem.h
 * @author Pascal Kraft
 * @brief This class contains a forward declaration of LocalProblem and NonLocalProblem and the class HierarchicalProblem
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <array>
#include <string>
#include <vector>
#include "../Core/Types.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include "../Core/FEDomain.h"
#include "../OutputGenerators/Images/ResidualOutputGenerator.h"

class LocalProblem;
class NonLocalProblem; 

/**
 * @brief The base class of the SweepingPreconditioner and general finite element system.
 * Since the object should call eachother recursively but the lowest level is different than the others, we use an abstract base class and two derived types.
 * 
 */
class HierarchicalProblem {
 public:
  /** @brief Direction in which this level sweeps over the problems of the level below. With three levels the levels 1, 2, 3 sweep in x, y, z; with two levels the levels 1, 2 sweep in y, z; with one level it is z. Has no meaning on level 0. */
  SweepingDirection sweeping_direction;
  /** @brief Level of this problem in the hierarchy. 0 is the local problem with the direct solver. */
  const SweepingLevel level;
  /** @brief Constraints of the dofs of this level in the global dof numbering of the level, built by make_constraints() and applied during assembly. */
  Constraints constraints;
  /** @brief Not used. It is never assigned or read. */
  std::array<dealii::IndexSet, 6> surface_index_sets;
  /** @brief Not used. It is never assigned or read. */
  std::array<bool, 6> is_hsie_surface;
  /** @brief Six entries, set to false in the constructor. Never read. */
  std:: vector<bool> is_surface_locked;
  /** @brief Not used. It is never assigned or read. */
  bool is_dof_manager_set;
  /** @brief True if level > 0, i.e. if there is a problem of the level below. */
  bool has_child;
  /** @brief The problem of the level below, which is used in the sweeping preconditioner of this level. Created by the constructor of NonLocalProblem; nullptr on level 0. */
  HierarchicalProblem* child = nullptr;
  /** @brief Sparsity pattern of the system matrix, built by make_sparsity_pattern(). */
  dealii:: SparsityPattern sp;
  /** @brief Solution of the last solve of this problem, in the global dof numbering of the level. */
  NumericVectorDistributed solution;
  /** @brief Not used. It is never assigned or read. */
  NumericVectorDistributed direct_solution;
  /** @brief Auxiliary vector with the layout of rhs. LocalProblem::assemble() stores a copy of the assembled right hand side in it. It is not read anywhere. */
  NumericVectorDistributed solution_error;
  /** @brief Right hand side of the system, in the global dof numbering of the level. On the levels below the top level the sweep of the parent overwrites it before every solve. */
  NumericVectorDistributed rhs;
  /** @brief Dofs of this level that are owned by this process (the entry of this process in LevelGeometry::dof_distribution). */
  dealii::IndexSet own_dofs;
  /** @brief Not used. It is never assigned or read. */
  std::array<std::vector<InterfaceDofData>, 6> surface_dof_associations;
  /** @brief System matrix of this level. Created in the constructors of the derived classes. */
  dealii::PETScWrappers::MPI::SparseMatrix * matrix = nullptr;
  /** @brief Names of the files written by the last call of output_results(): first the file of the inner domain, then one file per PML domain. */
  std::vector<std::string> filenames;
  /** @brief Stores the GMRES residuals of every solve of this problem and prints or plots them. Created by NonLocalProblem; nullptr on level 0. */
  ResidualOutputGenerator * residual_output = nullptr; // only used by problems that are solved iteratively
  /** @brief Number of calls of solve_with_timers_and_count() on this process. */
  unsigned int solve_counter;
  /** @brief Block index of this process in the sweeping direction of the level above. It identifies which of the problems of this level this process belongs to and is printed in front of the residuals. -1 if there is no level above. */
  int parent_sweeping_rank = -1;

  /**
   * @brief Construct a new Hierarchical Problem object
   * Inits the level member, stores the direction of the sweep and the solve counter.
   * 
   * @param level Level this problem describes.
   * @param direction The direction to sweep in. Doesnt matter for the LocalProblem.
   */
  HierarchicalProblem(unsigned int level, SweepingDirection direction);

  /**
   * @brief Not implemented on this level.
   * 
   */
  virtual ~HierarchicalProblem() =0;

  /**
   * @brief Not implemented on this level.
   * 
   */
  virtual void solve()=0;


  /**
   * @brief Not implemented on this level.
   * 
   */
  virtual void solve_adjoint() {};
  
  /**
   * @brief This function calls the objects solve() method but wraps a timer computation around it.
   * 
   */
  void solve_with_timers_and_count();
  
  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void initialize()=0;

  /**
   * @brief This function constructs all the required AffineConstraint objects. These couple the dofs in the inner domain and the boundary conditions together and is used for in-place condensation during matrix assembly.
   * 
   */
  void make_constraints();

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void assemble()=0;

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void initialize_index_sets()=0;

  /**
   * @brief For a given AffineConstraints object, this function adds constraints relating to numbering of dofs on two different structures.
   * 
   * This function can be used to couple boundary methods together or to couple dofs from a boundary method with dofs on the inner domain.
   * @param set_one First index set.
   * @param set_two Second index set.
   * @param affine_constraints Affine Constraint object to write the constraints into.
   */
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two, Constraints *affine_constraints);

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual auto reinit() -> void = 0;

  /**
   * @brief For a provided boundary id this returns the opposing one
   * The opposing sides are 0 and 1, 2 and 3, 4 and 5.
   * This function is usually required when a function should be called when all neighboring boundaries should be iterated. In that case we iterate from 0 to 5 and exclude the one we are currently on and the opposing one.
   * @return BoundaryId The BoundaryId of the opposing side.
   */
  auto opposing_site_bid(BoundaryId) -> BoundaryId;

  /**
   * @brief Computes a vector storing the difference between the precise rhs and the approximation by the solution.
   * This updates a vector called rhs_mismatch by filling it with the \f$A x - b\f$.
   * 
   */
  void compute_final_rhs_mismatch();

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void compute_solver_factorization() = 0;

  /**
   * @brief Basic functionality to write output files for a solution.
   * 
   * @param in_fname_part Core of the filename of the files.
   * @return std::string actually used filename with path which can be used to write meta data.
   */
  std::string output_results(std::string in_fname_part = "solution_inner_domain_level");

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void reinit_rhs() = 0;

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void make_sparsity_pattern() = 0;

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void update_convergence_criterion(double) {}

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   * @return unsigned int 
   */
  virtual unsigned int compute_global_solve_counter() {
    return 0;
  }

  /**
   * @brief This function uses the return values of compute_flobal_solve_counter to create some CLI output.
   * The function is recursive.
   */
  void print_solve_counter_list();

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   */
  virtual void empty_memory();

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   * @param filename 
   * @param apply_coordinate_transform 
   */
  virtual void write_multifile_output(const std::string & filename, bool apply_coordinate_transform) =0;

  /**
   * @brief Not implemented on this level, see derived classes.
   * 
   * @return std::vector<double> 
   */
  virtual std::vector<double> compute_shape_gradient() {
    return std::vector<double>();
  }
};

/**
 * @brief Context of the PETSc shell preconditioner of a NonLocalProblem.
 * PETSc passes it to the apply callback, which uses it to call NonLocalProblem::apply_sweep().
 */
typedef struct {
  /** @brief The problem whose sweeping preconditioner this context belongs to. */
  NonLocalProblem * parent;
} SampleShellPC;