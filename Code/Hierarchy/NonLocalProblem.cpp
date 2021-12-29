#include "NonLocalProblem.h"
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/staticfunctions.h"
#include "HierarchicalProblem.h"
#include "LocalProblem.h"
#include "../Core/InnerDomain.h"
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <mpi.h>
#include <petscksp.h>
#include <petscsystypes.h>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <ratio>
#include <string>
#include <vector>

static PetscErrorCode MonitorError(KSP , PetscInt its, PetscReal rnorm, void * problem)
{
  ((NonLocalProblem *)problem)->residual_output->push_value(rnorm);
  ((NonLocalProblem *)problem)->residual_output->write_residual_statement_to_console();
  ((NonLocalProblem *)problem)->child->update_convergence_criterion(rnorm);
  return(0);
}

int convergence_test(KSP , const PetscInt iteration, const PetscReal residual_norm, KSPConvergedReason *reason, void * solver_control_x)
{
  SolverControl &solver_control = *reinterpret_cast<SolverControl *>(solver_control_x);

  const SolverControl::State state = solver_control.check(iteration, residual_norm);

  switch (state)
    {
      case ::SolverControl::iterate:
        *reason = KSP_CONVERGED_ITERATING;
        break;

      case ::SolverControl::success:
        *reason = static_cast<KSPConvergedReason>(1);
        break;

      case ::SolverControl::failure:
        if (solver_control.last_step() > solver_control.max_steps())
          *reason = KSP_DIVERGED_ITS;
        else
          *reason = KSP_DIVERGED_DTOL;
        break;

      default:
        Assert(false, ExcNotImplemented());
    }

  // return without failure
  return 0;
}

void get_petsc_index_array_from_index_set(PetscInt* in_array, dealii::IndexSet in_set) {
  for(unsigned int i = 0; i < in_set.n_elements(); i++) {
    in_array[i] = in_set.nth_index_in_set(i);
  }
}

PetscErrorCode pc_apply(PC in_pc, Vec x_in, Vec x_out) {
  SampleShellPC  *shell;
  PCShellGetContext(in_pc,(void**)&shell);
  shell->parent->apply_sweep(x_in, x_out);
  return 0;
}

PetscErrorCode pc_create(SampleShellPC *shell, NonLocalProblem * parent)
{
  SampleShellPC  *newctx;
  PetscNew(&newctx);
  newctx->parent = parent;
  *shell = *newctx;
  return 0;
}

NonLocalProblem::NonLocalProblem(unsigned int level) :
  HierarchicalProblem(level, static_cast<SweepingDirection> (2 + GlobalParams.Sweeping_Level - level)),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true)
{
  sweeping_direction = get_sweeping_direction_for_level(level);
  if(level > 1) {
    child = new NonLocalProblem(level - 1);
  } else {
    child = new LocalProblem();
  }

  prepare_sweeping_data();

  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  
  locally_active_dofs = dealii::IndexSet(Geometry.levels[level].n_total_level_dofs);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->global_index_mapping.size(); i++) {
    locally_active_dofs.add_index(Geometry.levels[level].inner_domain->global_index_mapping[i]);
  }
  
  for(unsigned int surf = 0; surf < 6; surf++) {
    Geometry.levels[level].surfaces[surf]->print_dof_validation();
  }
  for(unsigned int surf = 0; surf < 6; surf++) {
    for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->global_index_mapping.size(); i++) {
      unsigned int global_index = Geometry.levels[level].surfaces[surf]->global_index_mapping[i];
      if(global_index > Geometry.levels[level].n_total_level_dofs) {
        
      } else {
        locally_active_dofs.add_index(global_index);
      }
    }  
  }
  n_locally_active_dofs = locally_active_dofs.n_elements();
  residual_output = new ResidualOutputGenerator("ConvergenceHistoryLevel"+std::to_string(level), "Convergence History on level " + std::to_string(level), total_rank_in_sweep, level , parent_sweeping_rank);
}

void NonLocalProblem::prepare_sweeping_data() {
  bool processed = false;

  if(sweeping_direction == SweepingDirection::X) {
    processed = true;
    n_blocks_in_sweeping_direction = GlobalParams.Blocks_in_x_direction;
    index_in_sweeping_direction = GlobalParams.Index_in_x_direction;
    total_rank_in_sweep = index_in_sweeping_direction;
    n_procs_in_sweep = GlobalParams.Blocks_in_x_direction;
    parent_sweeping_rank = GlobalParams.Index_in_y_direction;
  }
  if(sweeping_direction == SweepingDirection::Y) {
    processed = true;
    n_blocks_in_sweeping_direction = GlobalParams.Blocks_in_y_direction;
    index_in_sweeping_direction = GlobalParams.Index_in_y_direction;
    total_rank_in_sweep = index_in_sweeping_direction;
    if(GlobalParams.Blocks_in_x_direction > 1) {
      total_rank_in_sweep = index_in_sweeping_direction * GlobalParams.Blocks_in_x_direction +  GlobalParams.Index_in_x_direction;
    }
    n_procs_in_sweep = GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction;
    parent_sweeping_rank = GlobalParams.Index_in_z_direction;
  }
  if(sweeping_direction == SweepingDirection::Z) {
    processed = true;
    n_blocks_in_sweeping_direction = GlobalParams.Blocks_in_z_direction;
    index_in_sweeping_direction = GlobalParams.Index_in_z_direction;
    total_rank_in_sweep = GlobalParams.MPI_Rank;
    n_procs_in_sweep = GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction * GlobalParams.Blocks_in_z_direction;
  }
  if(!processed) {
    std::cout << "FAILURE on " << level << std::endl;
  } 
  // std::cout << "On level " << level << " and global rank " << GlobalParams.MPI_Rank << ": I am " << index_in_sweeping_direction << " and total rank " << total_rank_in_sweep << ". [" << GlobalParams.Index_in_x_direction << "x"<< GlobalParams.Index_in_y_direction << "x"<<GlobalParams.Index_in_z_direction << "]"<< std::endl;
}

void NonLocalProblem::init_solver_and_preconditioner() {
  // dealii::PETScWrappers::PreconditionNone pc_none;
  // pc_none.initialize(*matrix);
  KSPCreate(GlobalMPI.communicators_by_level[level], &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetOperators(ksp, *matrix, *matrix);
  KSPSetType(ksp, KSPGMRES);
  PCSetType(pc, PCSHELL);
  pc_create(&shell, this);
  PCShellSetApply(pc,pc_apply);
  PCShellSetContext(pc, (void*) &shell);
  KSPSetPC(ksp, pc);
  // KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc), nullptr);
  KSPSetPCSide(ksp, PCSide::PC_RIGHT);
  KSPGMRESSetRestart(ksp, GlobalParams.GMRES_max_steps);
  KSPMonitorSet(ksp, MonitorError, this, nullptr);
  KSPSetUp(ksp);
  KSPSetTolerances(ksp, 1e-10, GlobalParams.absolute_convergence_criterion, 1000, GlobalParams.GMRES_max_steps);
}

void NonLocalProblem::reinit_rhs() {
  reinit_vector(&rhs);
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
}

Direction get_direction_for_boundary_id(BoundaryId bid) {
  switch (bid)
  {
  case 0:
    return Direction::MinusX;
    break;
  case 1:
    return Direction::PlusX;
    break;
  case 2:
    return Direction::MinusY;
    break;
  case 3:
    return Direction::PlusY;
    break;
  case 4:
    return Direction::MinusZ;
    break;
  case 5:
    return Direction::PlusZ;
    break;
  
  default:
    std::cout << "Error in input for get_direction_for_boundary_id"<<std::endl;
    return Direction::MinusX;
    break;
  }
}

void NonLocalProblem::assemble() {
  print_info("NonLocalProblem::assemble", "Begin assembly");
  GlobalTimerManager.switch_context("Assemble", level);
  Timer timer;
  timer.start();
  Geometry.levels[level].inner_domain->assemble_system(&constraints, matrix, &rhs);
  print_info("NonLocalProblem::assemble", "Inner assembly done. Assembling boundary method contributions.");
  for(unsigned int i = 0; i< 6; i++) {
      Geometry.levels[level].surfaces[i]->fill_matrix(matrix, &rhs, &constraints);
  }
  timer.stop();
  print_info("NonLocalProblem::assemble", "Compress matrix.");
  matrix->compress(dealii::VectorOperation::add);
  rhs.compress(dealii::VectorOperation::add);
  print_info("NonLocalProblem::assemble", "Assemble child.");
  child->assemble();
  print_info("NonLocalProblem::assemble", "Compress vectors.");
  solution.compress(dealii::VectorOperation::add);
  rhs.compress(VectorOperation::add);
  // constraints.distribute(solution);
  print_info("NonLocalProblem::assemble", "End assembly.");
  GlobalTimerManager.leave_context(level);
}

void NonLocalProblem::solve() {
  
  if(level == GlobalParams.Sweeping_Level) {
    print_vector_norm(&rhs, "RHS");
  }

  bool run_itterative_solver = !GlobalParams.solve_directly;

  if(run_itterative_solver) {
    residual_output->new_series("Run " + std::to_string(solve_counter + 1));
    // Solve with sweeping
    
    PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);
    residual_output->close_current_series();
    if(ierr != 0) {
      std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    }

  } else {
    // Solve Directly for reference
    SolverControl sc;
    dealii::PETScWrappers::SparseDirectMUMPS direct_solver(sc, GlobalMPI.communicators_by_level[level]);
    direct_solver.solve(*matrix, solution, rhs);
  }
  

}

void NonLocalProblem::apply_sweep(Vec b_in, Vec u_out) {
  NumericVectorDistributed u = vector_from_vec_obj(b_in);
  // constraints.distribute(u);
  perform_downward_sweep( &u );
  perform_upward_sweep( &u );
  set_x_out_from_u(u_out, &u);
}

void NonLocalProblem::set_x_out_from_u(Vec x_out, NumericVectorDistributed * in_u) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  
  in_u->extract_subvector_to(vector_copy_own_indices, vector_copy_array);

  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
    values[i] = vector_copy_array[i];
  }
  
  VecSetValues(x_out, own_dofs.n_elements(), locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(x_out);
  VecAssemblyEnd(x_out);
  delete[] values;
}

void NonLocalProblem::S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst) {
  set_child_rhs_from_vector(src);
  child->solve_with_timers_and_count();
  set_vector_from_child_solution(dst);
}

void NonLocalProblem::set_vector_from_child_solution(NumericVectorDistributed * in_u) {
  child->solution.extract_subvector_to(vector_copy_child_indeces, vector_copy_array);
  in_u->set(vector_copy_own_indices, vector_copy_array);
  //in_u->compress(VectorOperation::insert);
}

void NonLocalProblem::set_child_rhs_from_vector(NumericVectorDistributed * in_u) {
  child->reinit_rhs();
  in_u->extract_subvector_to(vector_copy_own_indices, vector_copy_array);
  child->rhs.set(vector_copy_child_indeces, vector_copy_array);
  child->rhs.compress(VectorOperation::insert);
  if(GlobalParams.Index_in_z_direction && level == 1) {
    // child->constraints.distribute(child->rhs);
  }
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting for level " + std::to_string(level));
  MPI_Barrier(MPI_COMM_WORLD);
  GlobalTimerManager.switch_context("Reinit", level);
  
  make_constraints();
  
  make_sparsity_pattern();

  reinit_rhs();
  std::vector<unsigned int> local_rows;
  for(unsigned int p = 0; p < Geometry.levels[level].dof_distribution.size(); p++) {
    local_rows.push_back(Geometry.levels[level].dof_distribution[p].n_elements());
  }
  reinit_vector(&solution);
  reinit_vector(&direct_solution);
  reinit_vector(&solution_error);
  IndexSet all_dofs(Geometry.levels[level].n_total_level_dofs);
  all_dofs.add_range(0,Geometry.levels[level].n_total_level_dofs);
  matrix->reinit(Geometry.levels[level].dof_distribution[total_rank_in_sweep], Geometry.levels[level].dof_distribution[total_rank_in_sweep], sp, GlobalMPI.communicators_by_level[level]);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    if(Geometry.levels[level].inner_domain->is_dof_owned[i] && Geometry.levels[level-1].inner_domain->is_dof_owned[i]) {
      vector_copy_own_indices.push_back(Geometry.levels[level].inner_domain->global_index_mapping[i]);
      vector_copy_child_indeces.push_back(Geometry.levels[level-1].inner_domain->global_index_mapping[i]);
      vector_copy_array.push_back(ComplexNumber(0.0, 0.0));
    }
  }
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].surface_type[surf] == Geometry.levels[level-1].surface_type[surf]) {
      if(Geometry.levels[level].surfaces[surf]->dof_counter != Geometry.levels[level-1].surfaces[surf]->dof_counter) {
        complex_pml_domain_matching(surf);
      } else {
        for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->n_locally_active_dofs; i++) {
          if(Geometry.levels[level].surfaces[surf]->is_dof_owned[i] && Geometry.levels[level-1].surfaces[surf]->is_dof_owned[i]) {
            register_dof_copy_pair(Geometry.levels[level].surfaces[surf]->global_index_mapping[i], Geometry.levels[level-1].surfaces[surf]->global_index_mapping[i]);
          }
        }
      }
    }
  }
  GlobalTimerManager.leave_context(level);
  print_info("Nonlocal reinit", "Reinit done for level " + std::to_string(level));
}

void NonLocalProblem::register_dof_copy_pair(DofNumber own_index, DofNumber child_index) {
  vector_copy_own_indices.push_back(own_index);
  vector_copy_child_indeces.push_back(child_index);
  vector_copy_array.push_back(ComplexNumber(0.0, 0.0));
}

void NonLocalProblem::complex_pml_domain_matching(BoundaryId in_bid) {
  // always more dofs on the lower level
  dealii::IndexSet lower_is (Geometry.levels[level-1].n_total_level_dofs);
  dealii::IndexSet upper_is (Geometry.levels[level].n_total_level_dofs);
  unsigned int counter = 0;
  auto higher_cell = Geometry.levels[level].surfaces[in_bid]->dof_handler.begin();
  auto lower_cell = Geometry.levels[level-1].surfaces[in_bid]->dof_handler.begin();
  auto higher_end = Geometry.levels[level].surfaces[in_bid]->dof_handler.end();
  auto lower_end = Geometry.levels[level-1].surfaces[in_bid]->dof_handler.end();
  while(higher_cell != higher_end) {
    bool found = true;
    // first find the same cell in the child
    if(! ((higher_cell->center() - lower_cell->center()).norm() < FLOATING_PRECISION)) {
      while((higher_cell->center() - lower_cell->center()).norm() > FLOATING_PRECISION && lower_cell != lower_end) {
        lower_cell++;
      }
      if(lower_cell == lower_end) {
        lower_cell = Geometry.levels[level-1].surfaces[in_bid]->dof_handler.begin();
      }
      while((higher_cell->center() - lower_cell->center()).norm() > FLOATING_PRECISION && lower_cell != lower_end) {
        lower_cell++;
      }
      if(lower_cell == lower_end) {
        found = false;
        std::cout << "ERROR IN COMPLEX PML DOMAIN MATCHING" << std::endl;
      }
    }
    if(found) {
      // lower_cell and higher_cell point to the same cell on two different levels. Match the dofs.
      const unsigned int n_dofs_per_cell = Geometry.levels[level].surfaces[in_bid]->dof_handler.get_fe().dofs_per_cell;
      std::vector<DofNumber> lower_dofs(n_dofs_per_cell);
      std::vector<DofNumber> upper_dofs(n_dofs_per_cell);
      lower_cell->get_dof_indices(lower_dofs);
      std::sort(lower_dofs.begin(), lower_dofs.end());
      higher_cell->get_dof_indices(upper_dofs);
      std::sort(upper_dofs.begin(), upper_dofs.end());
      for(unsigned int i = 0; i < n_dofs_per_cell; i++) {
        if(Geometry.levels[level].surfaces[in_bid]->is_dof_owned[upper_dofs[i]] && Geometry.levels[level-1].surfaces[in_bid]->is_dof_owned[lower_dofs[i]]) {
          lower_is.add_index(Geometry.levels[level-1].surfaces[in_bid]->global_index_mapping[lower_dofs[i]]);
          upper_is.add_index(Geometry.levels[level].surfaces[in_bid]->global_index_mapping[upper_dofs[i]]);
        }
      }
    }
    lower_cell++;
    higher_cell++;
  }
  for(unsigned int i = 0; i < upper_is.n_elements(); i++) {
    register_dof_copy_pair(upper_is.nth_index_in_set(i), lower_is.nth_index_in_set(i));
  }
}

void NonLocalProblem::initialize() {
  GlobalTimerManager.switch_context("Initialize", level);
  child->initialize();
  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
  GlobalTimerManager.leave_context(level);
}
 
void NonLocalProblem::initialize_index_sets() {
  own_dofs = Geometry.levels[level].dof_distribution[total_rank_in_sweep];
  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
}

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
}

std::string NonLocalProblem::output_results() {
  print_info("NonLocalProblem", "Start output results on level" + std::to_string(level));
  print_solve_counter_list();
  update_shared_solution_vector();
  FEErrorStruct errors = compute_global_errors(&shared_solution);
  std::cout << "Errors: L2 = " << errors.L2 << " and Linfty  = " << errors.Linfty <<std::endl;
  write_multifile_output("solution", solution);
  ComplexNumber signal_strength = compute_signal_strength_of_solution();
  std::cout << "Signal strength: " << signal_strength << " with norm " << std::abs(signal_strength)<< std::endl;
  print_info("NonLocalProblem", "End output results on level" + std::to_string(level));
  return "";
}

void NonLocalProblem::update_shared_solution_vector() {
  shared_solution.reinit(own_dofs, locally_active_dofs, GlobalMPI.communicators_by_level[level]);
  for(unsigned int i= 0; i < own_dofs.n_elements(); i++) {
    shared_solution[own_dofs.nth_index_in_set(i)] = solution[own_dofs.nth_index_in_set(i)];
  }
  shared_solution.update_ghost_values();
}

void NonLocalProblem::write_multifile_output(const std::string & in_filename, const NumericVectorDistributed field) {
  if(GlobalParams.MPI_Rank == 0 && !GlobalParams.solve_directly) {
    residual_output->run_gnuplot();
    if(level > 1) {
      child->residual_output->run_gnuplot();
      if(level == 3) {
        child->child->residual_output->run_gnuplot();
      }  
    }
  }
  std::vector<std::string> generated_files;
    
  NumericVectorLocal local_solution(Geometry.levels[level].inner_domain->n_locally_active_dofs);
  
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    local_solution[i] = shared_solution[Geometry.levels[level].inner_domain->global_index_mapping[i]];
  }
  std::string file_1 = Geometry.levels[level].inner_domain->output_results(in_filename + std::to_string(level) , local_solution);
  generated_files.push_back(file_1);
  if(GlobalParams.BoundaryCondition == BoundaryConditionType::PML) {
    for (unsigned int surf = 0; surf < 6; surf++) {
      if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE){
        dealii::Vector<ComplexNumber> ds (Geometry.levels[level].surfaces[surf]->n_locally_active_dofs);
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[surf]->n_locally_active_dofs; index++) {
          ds[index] = shared_solution[Geometry.levels[level].surfaces[surf]->global_index_mapping[index]];
        }
        std::string file_2 = Geometry.levels[level].surfaces[surf]->output_results(ds, in_filename + "_pml" + std::to_string(level));
        generated_files.push_back(file_2);
      }
    }
  }
  std::vector<std::vector<std::string>> all_files = dealii::Utilities::MPI::gather(GlobalMPI.communicators_by_level[level], generated_files);
  if(GlobalParams.MPI_Rank == 0) {
    std::vector<std::string> flattened_filenames;
    for(unsigned int i = 0; i < all_files.size(); i++) {
      for(unsigned int j = 0; j < all_files[i].size(); j++) {
        flattened_filenames.push_back(all_files[i][j]);
      }
    }
    std::string filename = GlobalOutputManager.get_full_filename("_" + in_filename + ".pvtu");
    std::ofstream outputvtu(filename);
    for(unsigned int i = 0; i < flattened_filenames.size(); i++) {
      flattened_filenames[i] = "../" + flattened_filenames[i];
    }
    Geometry.levels[level].inner_domain->data_out.write_pvtu_record(outputvtu, flattened_filenames);
  }
}

void NonLocalProblem::communicate_external_dsp(DynamicSparsityPattern * in_dsp) {
  std::vector<std::vector<unsigned int>> rows, cols;
  for(unsigned int i = 0; i < n_procs_in_sweep; i++) {
    rows.emplace_back();
    cols.emplace_back();
  }
  for(auto it = in_dsp->begin(); it != in_dsp->end(); it++) {
    if(!own_dofs.is_element(it->row())) {
      for(unsigned int proc = 0; proc < n_procs_in_sweep; proc++) {
        if(Geometry.levels[level].dof_distribution[proc].is_element(it->row())) {
          rows[proc].push_back(it->row());
          cols[proc].push_back(it->column());
        }
      }
    }
  }
  unsigned int * entries_by_proc = new unsigned int[n_procs_in_sweep];
  for(unsigned int i = 0; i < n_procs_in_sweep; i++) {
    entries_by_proc[i] = rows[i].size();
  }
  unsigned int * recv_buffer = new unsigned int[n_procs_in_sweep];
  MPI_Alltoall(entries_by_proc, 1, MPI_UNSIGNED, recv_buffer, 1, MPI_UNSIGNED, GlobalMPI.communicators_by_level[level]);
  for(unsigned int other_proc = 0; other_proc < n_procs_in_sweep; other_proc++) {
    if(other_proc != total_rank_in_sweep) {
      if(recv_buffer[other_proc] != 0 || entries_by_proc[other_proc] != 0) {
        if(total_rank_in_sweep < other_proc) {
          // Send then receive
          if(entries_by_proc[other_proc] > 0) {
            unsigned int * sent_rows = new unsigned int [entries_by_proc[other_proc]];
            unsigned int * sent_cols = new unsigned int [entries_by_proc[other_proc]];
            for(unsigned int i = 0; i < entries_by_proc[other_proc]; i++) {
              sent_rows[i] = rows[other_proc][i];
              sent_cols[i] = cols[other_proc][i];
            }
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, total_rank_in_sweep, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, total_rank_in_sweep, GlobalMPI.communicators_by_level[level]);
            delete[] sent_rows;
            delete[] sent_cols;
          }
          // receive part
          if(recv_buffer[other_proc] > 0) {
            // There is something to receive
            unsigned int * received_rows = new unsigned int [recv_buffer[other_proc]];
            unsigned int * received_cols = new unsigned int [recv_buffer[other_proc]];
            MPI_Recv(received_rows, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, other_proc, GlobalMPI.communicators_by_level[level], 0);
            MPI_Recv(received_cols, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, other_proc, GlobalMPI.communicators_by_level[level], 0);
            for(unsigned int i = 0; i < recv_buffer[other_proc]; i++) {
              in_dsp->add(received_rows[i], received_cols[i]);
            }
            delete[] received_rows;
            delete[] received_cols;
          }
        } else {
          // Receive then send
          if(recv_buffer[other_proc] > 0) {
            // There is something to receive
            unsigned int * received_rows = new unsigned int [recv_buffer[other_proc]];
            unsigned int * received_cols = new unsigned int [recv_buffer[other_proc]];
            MPI_Recv(received_rows, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, other_proc, GlobalMPI.communicators_by_level[level], 0);
            MPI_Recv(received_cols, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, other_proc, GlobalMPI.communicators_by_level[level], 0);
            for(unsigned int i = 0; i < recv_buffer[other_proc]; i++) {
              in_dsp->add(received_rows[i], received_cols[i]);
            }
            delete[] received_cols;
            delete[] received_rows;
          }

          if(entries_by_proc[other_proc] > 0) {
            unsigned int * sent_rows = new unsigned int [entries_by_proc[other_proc]];
            unsigned int * sent_cols = new unsigned int [entries_by_proc[other_proc]];
            for(unsigned int i = 0; i < entries_by_proc[other_proc]; i++) {
              sent_rows[i] = rows[other_proc][i];
              sent_cols[i] = cols[other_proc][i];
            }
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, total_rank_in_sweep, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, total_rank_in_sweep, GlobalMPI.communicators_by_level[level]);
            delete[] sent_rows;
            delete[] sent_cols;
          }
        }
      }
    }
  }
  delete[] recv_buffer;
  delete[] entries_by_proc;
}

void NonLocalProblem::make_sparsity_pattern() {
  print_info("NonLocalProblem::make_sparsity_pattern", "Start on level "  + std::to_string(level));
  dealii::DynamicSparsityPattern dsp = {Geometry.levels[level].n_total_level_dofs, Geometry.levels[level].n_total_level_dofs};
  
  Geometry.levels[level].inner_domain->fill_sparsity_pattern(&dsp, &constraints);
  for (unsigned int surface = 0; surface < 6; surface++) {
    Geometry.levels[level].surfaces[surface]->fill_sparsity_pattern(&dsp, &constraints);
  }
  
  communicate_external_dsp(&dsp);
  sp.copy_from(dsp);
  sp.compress();
  print_info("NonLocalProblem::make_sparsity_pattern", "End on level "  + std::to_string(level));
}

NumericVectorDistributed NonLocalProblem::vector_from_vec_obj(Vec in_v) {
  NumericVectorDistributed ret;
  reinit_vector(&ret);
  const unsigned int n_loc_dofs = own_dofs.n_elements();
  const ComplexNumber * pointer;
  VecGetArrayRead(in_v, &pointer);
  for(unsigned int i = 0; i < n_loc_dofs; i++) {
    ret[own_dofs.nth_index_in_set(i)] = *pointer;
    pointer++;
  }
  VecRestoreArrayRead(in_v, &pointer);
  ret.compress(dealii::VectorOperation::insert);
  return ret;
}

void NonLocalProblem::print_vector_norm(NumericVectorDistributed * in_v, std::string marker) {
  in_v->extract_subvector_to(vector_copy_own_indices, vector_copy_array);
  double local_norm = 0.0;
  double max = 0;
  for(unsigned int i = 0; i < vector_copy_array.size(); i++) {
    double local = std::abs(vector_copy_array[i])*std::abs(vector_copy_array[i]);
    if(local > max) {
      max = local;
    } 
    local_norm += local;
  }
  local_norm = dealii::Utilities::MPI::sum(local_norm, GlobalMPI.communicators_by_level[level]);
  if(index_in_sweeping_direction == 0) {
    std::cout << marker << ": " << std::sqrt(local_norm) << std::endl;
  }
}

void NonLocalProblem::reinit_vector(NumericVectorDistributed * in_v) {
  in_v->reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
}

void NonLocalProblem::perform_downward_sweep(NumericVectorDistributed * u) {
  for(int i = n_blocks_in_sweeping_direction - 1; i >= 0; i--) {
    NumericVectorDistributed temp1, temp2;
    reinit_vector(&temp1);
    reinit_vector(&temp2);
    if(index_in_sweeping_direction == i) {
      S_inv(u, &temp1);
    }
    temp1.compress(VectorOperation::insert);
    matrix->vmult(temp2, temp1);
    if(index_in_sweeping_direction == i-1) {
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        ComplexNumber current_value((*u)(index).real(), (*u)(index).imag());
        ComplexNumber delta(temp2[index].real(), temp2[index].imag());
        (*u)[index] = current_value - delta;
      }
    }
    if(index_in_sweeping_direction == i) {
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        (*u)[index] = (ComplexNumber) temp1[index];
      }
    }
    u->compress(VectorOperation::insert);
  }
}

void NonLocalProblem::perform_upward_sweep(NumericVectorDistributed * in_u) {
  for(unsigned int i = 0; i < n_blocks_in_sweeping_direction-1; i++) {
    NumericVectorDistributed temp1, temp2, temp3;
    reinit_vector(&temp1);
    reinit_vector(&temp2);
    reinit_vector(&temp3);
    if(index_in_sweeping_direction == i) {
      for(unsigned int index = 0; index < own_dofs.n_elements(); index++) {
        temp1[own_dofs.nth_index_in_set(index)] = (ComplexNumber)((*in_u)[own_dofs.nth_index_in_set(index)]);
      }
    }
    temp1.compress(VectorOperation::insert);
    matrix->Tvmult(temp2, temp1);
    
    if(index_in_sweeping_direction == i+1) {
      S_inv(&temp2, &temp3);
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        ComplexNumber current_value = (*in_u)(index);
        ComplexNumber delta = temp3[index];
        (*in_u)[index] = current_value - delta;
      }   
    }
    in_u->compress(VectorOperation::insert);
  }
}

ComplexNumber NonLocalProblem::compute_signal_strength_of_solution() {
  print_info("NonLocalProblem::compute_signal_strength_of_solution", "Start");
  ComplexNumber integral = Geometry.levels[level].inner_domain->compute_signal_strength(& shared_solution);
  double sum_integral_real = dealii::Utilities::MPI::sum(integral.real(), GlobalMPI.communicators_by_level[level]);
  double sum_integral_imag = dealii::Utilities::MPI::sum(integral.imag(), GlobalMPI.communicators_by_level[level]);
  return ComplexNumber(sum_integral_real / (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction), sum_integral_imag / (GlobalParams.Blocks_in_x_direction * GlobalParams.Blocks_in_y_direction));
  print_info("NonLocalProblem::compute_signal_strength_of_solution", "End");
}

FEErrorStruct NonLocalProblem::compute_global_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution) {
  FEErrorStruct errors = Geometry.levels[level].inner_domain->compute_errors(in_solution);
  FEErrorStruct ret;
  ret.L2 = Utilities::MPI::sum(errors.L2, GlobalMPI.communicators_by_level[level]);
  ret.Linfty = Utilities::MPI::max(errors.Linfty, GlobalMPI.communicators_by_level[level]);
  return ret;
}

void NonLocalProblem::update_convergence_criterion(double last_residual) {
  if(GlobalParams.use_relative_convergence_criterion) {
    double base_value = last_residual;
    if(last_residual > 1.0) {
      base_value = 1.0;
    }
    double new_abort_limit = base_value * GlobalParams.relative_convergence_criterion;
    new_abort_limit = std::max(new_abort_limit, GlobalParams.absolute_convergence_criterion);
    KSPSetTolerances(ksp, 1e-10,new_abort_limit , 1000, GlobalParams.GMRES_max_steps);
    // std::cout << "Setting level " << level << " convergence criterion to " << new_abort_limit << std::endl;
  }
}

unsigned int NonLocalProblem::compute_global_solve_counter() {
  unsigned int contribution = 0;
  if(total_rank_in_sweep == 0) {
    contribution = solve_counter;
  }
  return Utilities::MPI::sum(contribution, MPI_COMM_WORLD);
}