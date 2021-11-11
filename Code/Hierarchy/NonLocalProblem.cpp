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

static PetscErrorCode MonitorError(KSP , PetscInt its, PetscReal rnorm, void * outputter)
{
  ((ResidualOutputGenerator *)outputter)->push_value(rnorm);
  ((ResidualOutputGenerator *)outputter)->write_residual_statement_to_console();
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
  HierarchicalProblem(level, static_cast<SweepingDirection> (2 + GlobalParams.HSIE_SWEEPING_LEVEL - level)),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true)
{
  if(level > 1) {
    child = new NonLocalProblem(level - 1);
  } else {
    child = new LocalProblem();
  }
  
  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  if(sweeping_direction == SweepingDirection::X) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_x_direction;
    index_in_sweep = GlobalParams.Index_in_x_direction;
  }
  if(sweeping_direction == SweepingDirection::Y) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_y_direction;
    index_in_sweep = GlobalParams.Index_in_y_direction;
  }
  if(sweeping_direction == SweepingDirection::Z) {
    n_blocks_in_sweep = GlobalParams.Blocks_in_z_direction;
    index_in_sweep = GlobalParams.Index_in_z_direction;
  }
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
  residual_output = new ResidualOutputGenerator("ConvergenceHistoryLevel"+std::to_string(level), "Convergence History on level " + std::to_string(level), index_in_sweep == 0, level );
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
  constraints.distribute(solution);
  print_info("NonLocalProblem::assemble", "End assembly.");
  if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Dirichlet) {
    for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
      if(constraints.is_inhomogeneously_constrained(own_dofs.nth_index_in_set(i))) {
        if(std::abs(constraints.get_inhomogeneity(own_dofs.nth_index_in_set(i))) > internal_vector_norm) {
          internal_vector_norm = std::abs(constraints.get_inhomogeneity(own_dofs.nth_index_in_set(i)));
        }
      }
    }
    internal_vector_norm = dealii::Utilities::MPI::max(internal_vector_norm, MPI_COMM_WORLD);
  }
  GlobalTimerManager.leave_context(level);
}

void NonLocalProblem::solve() {
  GlobalTimerManager.switch_context("Solve", level);
  
  constraints.distribute(solution);
  if(level == GlobalParams.HSIE_SWEEPING_LEVEL) {
    print_vector_norm(&rhs, "RHS");
  }
  
  bool run_itterative_solver = !GlobalParams.solve_directly;
  // if(level == 1) run_itterative_solver = false;

  if(run_itterative_solver) {
    residual_output->new_series("Run " + std::to_string(solve_counter + 1));
    // Solve with sweeping
    KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc), nullptr);
    KSPSetPCSide(ksp, PCSide::PC_RIGHT);
    KSPGMRESSetRestart(ksp, GlobalParams.GMRES_max_steps);
    KSPSetTolerances(ksp, 0.0001, 1.0, 1000, GlobalParams.GMRES_max_steps);
    KSPMonitorSet(ksp, MonitorError, residual_output, nullptr);
    KSPSetUp(ksp);
    PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);
    residual_output->close_current_series();
    if(ierr != 0) {
      std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    //   throw new ExcPETScError(ierr);
    }  
  } else {
    // Solve Directly for reference
    SolverControl sc;
    dealii::PETScWrappers::SparseDirectMUMPS solver1(sc, GlobalMPI.communicators_by_level[level]);
    solver1.solve(*matrix, solution, rhs);
  }
  GlobalTimerManager.leave_context(level);
  // subtract_vectors(&solution, &solution_error);
  // constraints.distribute(solution);

  solve_counter++;
  if(level == GlobalParams.HSIE_SWEEPING_LEVEL) {
    matrix->residual(solution_error, solution, rhs);
    write_multifile_output("error_of_solution", solution_error);
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
  double norm = 0;
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
    values[i] = vector_copy_array[i];
    norm += std::abs(values[i]) * std::abs(values[i]);
  }
  
  VecSetValues(x_out, own_dofs.n_elements(), locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(x_out);
  VecAssemblyEnd(x_out);
  delete[] values;
}

void NonLocalProblem::S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst) {
  set_child_rhs_from_vector(src);
  child->solve();
  set_vector_from_child_solution(dst);
}

void NonLocalProblem::set_vector_from_child_solution(NumericVectorDistributed * in_u) {
  child->solution.extract_subvector_to(vector_copy_child_indeces, vector_copy_array);
  double norm = 0;
  for(unsigned int i = 0; i < vector_copy_array.size(); i++) {
    norm += std::abs(vector_copy_array[i])*std::abs(vector_copy_array[i]);
  }
  norm = std::sqrt(norm);
  // std::cout << "Copied norm: " << norm << std::endl;
  in_u->set(vector_copy_own_indices, vector_copy_array);
}

void NonLocalProblem::set_child_rhs_from_vector(NumericVectorDistributed * in_u) {
  child->reinit_rhs();
  in_u->extract_subvector_to(vector_copy_own_indices, vector_copy_array);
  child->rhs.set(vector_copy_child_indeces, vector_copy_array);
  child->rhs.compress(VectorOperation::insert);
  if(GlobalParams.Index_in_z_direction && level == 1) {
    child->constraints.distribute(child->rhs);
  }
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting for level " + std::to_string(level));
  MPI_Barrier(MPI_COMM_WORLD);
  GlobalTimerManager.switch_context("Reinit", level);
  // child->reinit();
  
  make_constraints();
  // print_diagnosis_data();
  
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
  matrix->reinit(Geometry.levels[level].dof_distribution[rank], Geometry.levels[level].dof_distribution[rank], sp, GlobalMPI.communicators_by_level[level]);
  //matrix->reinit(GlobalMPI.communicators_by_level[level], sp,local_rows ,local_rows, rank);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    if(Geometry.levels[level].inner_domain->is_dof_owned[i] && Geometry.levels[level-1].inner_domain->is_dof_owned[i]) {
      vector_copy_own_indices.push_back(Geometry.levels[level].inner_domain->global_index_mapping[i]);
      vector_copy_child_indeces.push_back(Geometry.levels[level-1].inner_domain->global_index_mapping[i]);
      vector_copy_array.push_back(ComplexNumber(0.0, 0.0));
    }
  }
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(Geometry.levels[level].surface_type[surf] == Geometry.levels[level-1].surface_type[surf]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->n_locally_active_dofs; i++) {
        if(Geometry.levels[level].surfaces[surf]->is_dof_owned[i] && Geometry.levels[level-1].surfaces[surf]->is_dof_owned[i]) {
          vector_copy_own_indices.push_back(Geometry.levels[level].surfaces[surf]->global_index_mapping[i]);
          vector_copy_child_indeces.push_back(Geometry.levels[level-1].surfaces[surf]->global_index_mapping[i]);
          vector_copy_array.push_back(ComplexNumber(0.0, 0.0));
        }
      }
    }
  }
  GlobalTimerManager.leave_context(level);
  print_info("Nonlocal reinit", "Reinit done for level " + std::to_string(level));
}

void NonLocalProblem::initialize() {
  GlobalTimerManager.switch_context("Initialize", level);
  child->initialize();
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(GlobalMPI.communicators_by_level[level]);
  rank = dealii::Utilities::MPI::this_mpi_process(GlobalMPI.communicators_by_level[level]);
  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
  GlobalTimerManager.leave_context(level);
}
 
void NonLocalProblem::initialize_index_sets() {
  own_dofs = Geometry.levels[level].dof_distribution[GlobalMPI.rank_on_level[level]];
  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
}

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
}

std::string NonLocalProblem::output_results() {
  print_info("NonLocalProblem", "Start output results on level" + std::to_string(level));
  write_multifile_output("solution", solution);
  print_info("NonLocalProblem", "End output results on level" + std::to_string(level));
  return "";
}

void NonLocalProblem::write_multifile_output(const std::string & in_filename, const NumericVectorDistributed field) {
  if(GlobalParams.MPI_Rank == 0 && !GlobalParams.solve_directly) {
    residual_output->run_gnuplot();
  }
  std::vector<std::string> generated_files;
  dealii::LinearAlgebra::distributed::Vector<ComplexNumber> shared_solution;
  shared_solution.reinit(own_dofs, locally_active_dofs, GlobalMPI.communicators_by_level[level]);
  for(unsigned int i= 0; i < own_dofs.n_elements(); i++) {
    shared_solution[own_dofs.nth_index_in_set(i)] = field[own_dofs.nth_index_in_set(i)];
  }
  shared_solution.update_ghost_values();
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
  const unsigned int n_procs = Geometry.levels[level].dof_distribution.size();
  for(unsigned int i = 0; i < n_procs; i++) {
    rows.emplace_back();
    cols.emplace_back();
  }
  for(auto it = in_dsp->begin(); it != in_dsp->end(); it++) {
    if(!own_dofs.is_element(it->row())) {
      for(unsigned int proc = 0; proc < n_procs; proc++) {
        if(Geometry.levels[level].dof_distribution[proc].is_element(it->row())) {
          rows[proc].push_back(it->row());
          cols[proc].push_back(it->column());
        }
      }
    }
  }
  unsigned int * entries_by_proc = new unsigned int[n_procs];
  for(unsigned int i = 0; i < n_procs; i++) {
    entries_by_proc[i] = rows[i].size();
  }
  unsigned int * recv_buffer = new unsigned int[n_procs];
  MPI_Alltoall(entries_by_proc, 1, MPI_UNSIGNED, recv_buffer, 1, MPI_UNSIGNED, GlobalMPI.communicators_by_level[level]);
  for(unsigned int other_proc = 0; other_proc < n_procs; other_proc++) {
    if(other_proc != rank) {
      if(recv_buffer[other_proc] != 0 || entries_by_proc[other_proc] != 0) {
        if(rank < other_proc) {
          // Send then receive
          if(entries_by_proc[other_proc] > 0) {
            unsigned int * sent_rows = new unsigned int [entries_by_proc[other_proc]];
            unsigned int * sent_cols = new unsigned int [entries_by_proc[other_proc]];
            for(unsigned int i = 0; i < entries_by_proc[other_proc]; i++) {
              sent_rows[i] = rows[other_proc][i];
              sent_cols[i] = cols[other_proc][i];
            }
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, rank, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, rank, GlobalMPI.communicators_by_level[level]);
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
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, rank, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, rank, GlobalMPI.communicators_by_level[level]);
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
  // std::cout << "on rank " << GlobalParams.MPI_Rank << " : " << local_norm << std::endl;
  local_norm = dealii::Utilities::MPI::sum(local_norm, GlobalMPI.communicators_by_level[level]);
  if(rank == 0) {
    std::cout << marker << " on " << GlobalParams.MPI_Rank << ": " << std::sqrt(local_norm) << " and " << max << std::endl;
  }
}

void NonLocalProblem::reinit_vector(NumericVectorDistributed * in_v) {
  in_v->reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
}

void NonLocalProblem::perform_downward_sweep(NumericVectorDistributed * u) {
  for(int i = n_procs_in_sweep - 1; i >= 0; i--) {
    NumericVectorDistributed temp1, temp2;
    reinit_vector(&temp1);
    reinit_vector(&temp2);
    if(index_in_sweep == i) {
      S_inv(u, &temp1);
    }
    temp1.compress(VectorOperation::insert);
    matrix->vmult(temp2, temp1);
    if(index_in_sweep == i-1) {
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        ComplexNumber current_value((*u)(index).real(), (*u)(index).imag());
        ComplexNumber delta(temp2[index].real(), temp2[index].imag());
        (*u)[index] = current_value - delta;
      }
    }
    if(index_in_sweep == i) {
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        (*u)[index] = (ComplexNumber) temp1[index];
      }
    }
    u->compress(VectorOperation::insert);
  }
  // print_vector_norm(u, "DownwardNorm");
}

void NonLocalProblem::perform_upward_sweep(NumericVectorDistributed * u) {
  for(unsigned int i = 0; i < n_procs_in_sweep-1; i++) {
    NumericVectorDistributed temp1, temp2, temp3;
    reinit_vector(&temp1);
    reinit_vector(&temp2);
    reinit_vector(&temp3);
    if(index_in_sweep == i) {
      for(unsigned int index = 0; index < own_dofs.n_elements(); index++) {
        temp1[own_dofs.nth_index_in_set(index)] = (ComplexNumber)((*u)[own_dofs.nth_index_in_set(index)]);
      }
    }
    temp1.compress(VectorOperation::insert);
    matrix->Tvmult(temp2, temp1);
    
    if(index_in_sweep == i+1) {
      S_inv(&temp2, &temp3);
      for(unsigned int j = 0; j < own_dofs.n_elements(); j++) {
        const unsigned int index = own_dofs.nth_index_in_set(j);
        ComplexNumber current_value = (*u)(index);
        ComplexNumber delta = temp3[index];
        (*u)[index] = current_value - delta;
      }   
    }
    u->compress(VectorOperation::insert);
  }
  // print_vector_norm(u, "UpwardNorm");
}