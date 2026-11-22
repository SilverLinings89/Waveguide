#include "../Core/Types.h"
#include "LocalProblem.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include "../Helpers/staticfunctions.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <deal.II/numerics/vector_tools_project.templates.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/matrix_free.templates.h>
#include <mpi.h>
#include "../Helpers/PointSourceField.h"
#include "../Solutions/ExactSolutionRamped.h"
#include "../Solutions/ExactSolutionConjugate.h"


template void dealii::VectorTools::project<3,dealii::Vector<ComplexNumber>,3>(const dealii::Mapping<3, 3> &, const dealii::DoFHandler<3, 3> &, const Constraints &,
        const dealii::Quadrature<3> &,
        const dealii::Function<3, ComplexNumber> &,
        dealii::Vector<ComplexNumber> &,
        const bool,
        const dealii::Quadrature<2> &,
        const bool);

LocalProblem::LocalProblem() :
  HierarchicalProblem(0, SweepingDirection::Z), 
  local_ksp(nullptr) {
    print_info("Local Problem", "Done building base problem. Preparing matrix.");
    matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
    for(unsigned int i = 0; i < 6; i++) Geometry.levels[0].is_surface_truncated[i] = true;
    own_dofs = Geometry.levels[0].dof_distribution[0];
}

LocalProblem::~LocalProblem() {
  if(local_ksp != nullptr) {
    KSPDestroy(&local_ksp);
  }
  delete matrix;
}

dealii::IndexSet LocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  dealii::IndexSet ret(Geometry.levels[0].n_local_dofs);
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      std::vector<InterfaceDofData> current = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(interface_id);
      for(unsigned int j = 0; j < current.size(); j++) {
        ret.add_index(current[j].index);
      }      
    } else {
      if(i != opposing_interface_id && Geometry.levels[0].is_surface_truncated[i]) {
        std::vector<InterfaceDofData> current = Geometry.levels[0].surfaces[i]->get_dof_association_by_boundary_id(i);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index);
        }
      }
    }
  }
  return ret;
}

void LocalProblem::initialize() {
  print_info("LocalProblem::initialize", "Start");
  // GlobalTimerManager.switch_context("Initialize", 0);
  print_info("LocalProblem::initialize", "Number of local dofs: " + std::to_string(Geometry.levels[0].n_local_dofs), LoggingLevel::DEBUG_ALL);
  reinit();
  // GlobalTimerManager.leave_context(0);
  print_info("LocalProblem::initialize", "End");
}

void LocalProblem::validate() {
  print_info("LocalProblem::validate", "Start");
  print_info("LocalProblem::validate", "Matrix size: (" + std::to_string(matrix->m()) + " x "  + std::to_string(matrix->n()) + ") and l1-norm " + std::to_string(matrix->l1_norm()), LoggingLevel::PRODUCTION_ONE);
  print_info("LocalProblem::validate", "End");
}

void LocalProblem::assemble() {
  GlobalTimerManager.switch_context("Assemble", level);
  matrix->operator=(0);
  has_factorization = false;
  rhs = 0;
  matrix->compress(dealii::VectorOperation::insert);
  rhs.compress(dealii::VectorOperation::insert);
  reinit_rhs();
  Timer timer;
  timer.start();
  Geometry.levels[level].inner_domain->assemble_system(&constraints, matrix, &rhs);
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(Geometry.levels[0].is_surface_truncated[surface]) {
      Geometry.levels[0].surfaces[surface]->fill_matrix(matrix, &rhs, &constraints);
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs.compress(dealii::VectorOperation::add);
  solution_error = rhs;
  timer.stop();
  GlobalTimerManager.leave_context(0);
}

void LocalProblem::reinit_rhs() {
  rhs.reinit(own_dofs, MPI_COMM_SELF);
}

void LocalProblem::make_sparsity_pattern() {
  print_info("LocalProblem::make_sparsity_pattern", "Start on the local level");
  dealii::DynamicSparsityPattern dsp = {Geometry.levels[0].n_total_level_dofs, Geometry.levels[0].n_total_level_dofs};
  
  Geometry.levels[0].inner_domain->fill_sparsity_pattern(&dsp, &constraints);
  for (unsigned int surface = 0; surface < 6; surface++) {
    Geometry.levels[0].surfaces[surface]->fill_sparsity_pattern(&dsp, &constraints);
  }
  
  sp.copy_from(dsp);
  sp.compress();
  print_info("LocalProblem::make_sparsity_pattern", "End on local level");
}

void LocalProblem::reinit() {
  GlobalTimerManager.switch_context("Reinit", 0);
  reinit_rhs();
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, MPI_COMM_SELF);
  solution.reinit(MPI_COMM_SELF, Geometry.levels[0].n_local_dofs, Geometry.levels[0].n_local_dofs, false);
  solution_error.reinit(MPI_COMM_SELF, Geometry.levels[0].n_local_dofs, Geometry.levels[0].n_local_dofs, false);
  solution = 0;
  solution_error = 0;
  make_constraints();
  constraints.close();
  make_sparsity_pattern();
  std::vector<unsigned int> lines_per_proc;
  lines_per_proc.push_back(sp.n_cols());
  matrix->reinit(MPI_COMM_SELF, sp, lines_per_proc, lines_per_proc, 0);
  GlobalTimerManager.leave_context(0);
}

void LocalProblem::solve() {
  rhs.compress(dealii::VectorOperation::insert);
  if(local_ksp == nullptr) {
    KSPCreate(PETSC_COMM_SELF, &local_ksp);
    KSPSetType(local_ksp, KSPPREONLY);
    PC local_pc;
    KSPGetPC(local_ksp, &local_pc);
    // The system matrix is complex symmetric (not hermitian), so an LDL^T factorization can be used.
    PCSetType(local_pc, PCCHOLESKY);
    PCFactorSetMatSolverType(local_pc, MATSOLVERMUMPS);
    KSPSetOptionsPrefix(local_ksp, "local_");
    KSPSetOperators(local_ksp, *matrix, *matrix);
    if(GlobalParams.local_solver_compression_tolerance > 0) {
      // Block low rank compression: the fronts of the multifrontal factorization are stored as low rank blocks
      // that are accurate up to the given tolerance. ICNTL(35) = 2 uses the compressed factors in the factorization
      // and in the solve phase, CNTL(7) is the tolerance.
      PCFactorSetUpMatSolverType(local_pc);
      Mat factor_matrix;
      PCFactorGetMatrix(local_pc, &factor_matrix);
      MatMumpsSetIcntl(factor_matrix, 35, 2);
      MatMumpsSetCntl(factor_matrix, 7, GlobalParams.local_solver_compression_tolerance);
    }
    // Options from the command line take precedence over the settings above.
    KSPSetFromOptions(local_ksp);
  }
  // PETSc keeps the factorization as long as the matrix is unchanged and computes a new one after the matrix has
  // been assembled again (for example for the next shape in an optimization run).
  KSPSetOperators(local_ksp, *matrix, *matrix);
  PC pc_for_statistics;
  KSPGetPC(local_ksp, &pc_for_statistics);
  // has_factorization is reset whenever the matrix is assembled again.
  const bool computes_factorization = !has_factorization;
  dealii::Timer solve_timer;
  solve_timer.start();
  PetscErrorCode ierr = KSPSolve(local_ksp, rhs, solution);
  solve_timer.stop();
  if(computes_factorization) {
    time_of_factorization += solve_timer.wall_time();
    has_factorization = true;
    Mat factor_matrix;
    PetscBool is_mumps = PETSC_FALSE;
    PCFactorGetMatrix(pc_for_statistics, &factor_matrix);
    MatSolverType solver_type;
    MatFactorGetSolverType(factor_matrix, &solver_type);
    PetscStrcmp(solver_type, MATSOLVERMUMPS, &is_mumps);
    if(is_mumps) {
      PetscInt value;
      // Negative values of these entries are counts in millions.
      MatMumpsGetInfog(factor_matrix, 29, &value);
      factor_entries_full_rank = (value < 0) ? -1e6 * (double) value : (double) value;
      MatMumpsGetInfog(factor_matrix, 35, &value);
      factor_entries_stored = (value < 0) ? -1e6 * (double) value : (double) value;
      MatMumpsGetInfog(factor_matrix, 22, &value);
      factor_memory_mb = (double) value;
    }
  } else {
    time_of_solves += solve_timer.wall_time();
    n_timed_solves++;
  }
  KSPConvergedReason reason;
  KSPGetConvergedReason(local_ksp, &reason);
  if(ierr != 0 || reason < 0) {
    throw std::runtime_error("LocalProblem::solve: The direct solver of the subdomain failed (PETSc converged reason " + std::to_string(reason) + "). The factorization may have run out of memory or the matrix is singular.");
  }
}

void LocalProblem::initialize_index_sets() {
  
}

void LocalProblem::compute_solver_factorization() {
  Timer timer1;
  print_info("LocalProblem::compute_solver_factorization", "Begin solver factorization: ", LoggingLevel::PRODUCTION_ONE);
  timer1.start();
  solve();
  timer1.stop();
  solution = 0;
  print_info("LocalProblem::compute_solver_factorization", "Walltime: " + std::to_string(timer1.wall_time()) , LoggingLevel::PRODUCTION_ONE);
}

double LocalProblem::compute_error() {
  Timer timer;
  timer.start ();
  double error = compute_L2_error();
  timer.stop ();
  print_info("LocalProblem::compute_error", "L2 Error: " + std::to_string(error) + " (computed in " + std::to_string(timer.cpu_time()) + "s)");
  return error;
}

double LocalProblem::compute_L2_error() {
  NumericVectorLocal solution_inner(Geometry.levels[level].inner_domain->n_locally_active_dofs);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    solution_inner[i] = solution(i);
  }
  dealii::Vector<double> cellwise_error(Geometry.levels[level].inner_domain->triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    Geometry.levels[level].inner_domain->dof_handler,
    solution_inner,
    *GlobalParams.source_field,
    cellwise_error,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  return dealii::VectorTools::compute_global_error(Geometry.levels[level].inner_domain->triangulation, cellwise_error, dealii::VectorTools::NormType::L2_norm);
}

unsigned int LocalProblem::compute_global_solve_counter() {
  return Utilities::MPI::sum(solve_counter, MPI_COMM_WORLD);
}

void LocalProblem::empty_memory() {
  matrix->clear();
  if(local_ksp != nullptr) {
    KSPDestroy(&local_ksp);
    local_ksp = nullptr;
  }
}

void LocalProblem::write_multifile_output(const std::string & in_filename, bool) {
  NumericVectorLocal local_solution(Geometry.levels[0].inner_domain->n_locally_active_dofs);
  std::vector<std::string> generated_files;
  for(unsigned int i = 0; i < Geometry.levels[0].inner_domain->n_locally_active_dofs; i++) {
    local_solution[i] = solution[i];
  }

  std::string file_1 = Geometry.levels[0].inner_domain->output_results(in_filename + "0" , local_solution, false);
  generated_files.push_back(file_1);
  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for (unsigned int surf = 0; surf < 6; surf++) {
      if(Geometry.levels[0].surface_type[surf] == SurfaceType::ABC_SURFACE){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[0].surfaces[surf]->n_locally_active_dofs);
        for(unsigned int index = 0; index < Geometry.levels[0].surfaces[surf]->n_locally_active_dofs; index++) {
          ds[index] = solution[Geometry.levels[0].surfaces[surf]->global_index_mapping[index]];
        }
        std::string file_2 = Geometry.levels[0].surfaces[surf]->output_results(ds, in_filename + "_pml0");
        generated_files.push_back(file_2);
      }
    }
  }

  std::string filename = GlobalOutputManager.get_full_filename("_" + in_filename + ".pvtu");
  std::ofstream outputvtu(filename);
  for(unsigned int i = 0; i < generated_files.size(); i++) {
    // The pvtu record is stored in the same folder as the pieces, so only the file name is referenced.
    generated_files[i] = generated_files[i].substr(generated_files[i].find_last_of('/') + 1);
  }
  Geometry.levels[0].inner_domain->data_out.write_pvtu_record(outputvtu, generated_files);
}
