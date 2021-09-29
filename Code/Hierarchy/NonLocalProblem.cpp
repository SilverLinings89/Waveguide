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

void print_dof_ranges(std::string name, FEDomain * in_fedomain) {
  std::cout << "For " << name << " I found the dofs ";
  for(unsigned int i = 0; i < in_fedomain->global_index_mapping.size(); i++) {
    std::cout << in_fedomain->global_index_mapping[i] << " "; 
  }
  std::cout << std::endl;
}

double l2_norm(DofFieldTrace in_trace) {
  double ret = 0;
  for(unsigned int i = 0; i < in_trace.size(); i++) {
    ret += in_trace[i].real()*in_trace[i].real() + in_trace[i].imag() * in_trace[i].imag();
  }
  return std::sqrt(ret);
}

double l2_norm(NumericVectorLocal in_vector) {
  double ret = 0;
  for(unsigned int i = 0; i < in_vector.size(); i++) {
    ret += in_vector[i].real()*in_vector[i].real() + in_vector[i].imag() * in_vector[i].imag();
  }
  return std::sqrt(ret);
}

static double last_residual = -1;

Direction get_lower_boundary_id_for_sweeping_direction(SweepingDirection in_direction) {
  if(in_direction == SweepingDirection::X) {
    return Direction::MinusX;
  }
  if(in_direction == SweepingDirection::Y) {
    return Direction::MinusY;
  }
  return Direction::MinusZ;
}

Direction get_upper_boundary_id_for_sweeping_direction(SweepingDirection in_direction) {
  if(in_direction == SweepingDirection::X) {
    return Direction::PlusX;
  }
  if(in_direction == SweepingDirection::Y) {
    return Direction::PlusY;
  }
  return Direction::PlusZ;
}

static PetscErrorCode MonitorError(KSP , PetscInt its, PetscReal rnorm, void *)
{
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout << "Residual in step " << std::to_string(its) << "  : " << std::to_string(rnorm);
    if(last_residual != -1 ) {
      std::cout << " Factor: " << std::to_string(rnorm / last_residual);
    }
    std::cout << std::endl;
    last_residual = rnorm;
  }
  return(0);
}

double l2_norm_of_vector(std::vector<ComplexNumber> input) {
  double norm = 0;
  for(unsigned int i = 0; i< input.size(); i++) {
    norm += std::abs(input[i]);
  }
  return std::sqrt(norm);
}

int convergence_test(KSP , const PetscInt iteration, const PetscReal residual_norm, KSPConvergedReason *reason, void * solver_control_x)
{
  SolverControl &solver_control =
    *reinterpret_cast<SolverControl *>(solver_control_x);

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
    for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surf]->global_index_mapping.size(); i++) {
      locally_active_dofs.add_index(Geometry.levels[level].surfaces[surf]->global_index_mapping[i]);
    }  
  }
  n_locally_active_dofs = locally_active_dofs.n_elements();
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
  rhs.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
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
  GlobalTimerManager.switch_context("assemble", level);
  Geometry.levels[level].inner_domain->assemble_system(&constraints, matrix, &rhs);
  print_info("NonLocalProblem::assemble", "Inner assembly done. Assembling boundary method contributions.");
  for(unsigned int i = 0; i< 6; i++) {
      Timer timer;
      timer.start();
      Geometry.levels[level].surfaces[i]->fill_matrix(matrix, &rhs, &constraints);
      timer.stop();
  }
  print_info("NonLocalProblem::assemble", "Compress matrix.");
  matrix->compress(dealii::VectorOperation::add);
  print_info("NonLocalProblem::assemble", "Assemble child.");
  child->assemble();
  print_info("NonLocalProblem::assemble", "Compress vectors.");
  solution.compress(dealii::VectorOperation::add);
  print_info("NonLocalProblem::assemble", "End assembly.");
  
}

void NonLocalProblem::solve() {
  init_solver_and_preconditioner();
  print_info("NonLocalProblem::solve", "Start");
  GlobalTimerManager.switch_context("solve", level);
  rhs.compress(VectorOperation::add);
  print_vector_norm(&rhs, "RHS");
  if(!GlobalParams.solve_directly) {
    // Solve with sweeping
    KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc), nullptr);
    KSPSetPCSide(ksp, PCSide::PC_RIGHT);
    KSPSetTolerances(ksp, 0.000001, 1.0, 1000, GlobalParams.GMRES_max_steps);
    KSPMonitorSet(ksp, MonitorError, nullptr, nullptr);
    KSPSetUp(ksp);
    PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);
    if(ierr != 0) {
      std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    //   throw new ExcPETScError(ierr);
    }  
  } else {
    // Solve Directly for reference
    SolverControl sc;
    dealii::PETScWrappers::SparseDirectMUMPS solver1(sc, MPI_COMM_WORLD);
    solver1.solve(*matrix, solution_error, rhs);
  }
  matrix->residual(solution_error, solution, rhs);
  // subtract_vectors(&solution, &solution_error);
  constraints.distribute(solution);
  write_multifile_output("error_of_solution", solution_error);
  print_info("NonLocalProblem::solve", "End");
}

void NonLocalProblem::apply_sweep(Vec b_in, Vec u_out) {
  NumericVectorDistributed temp_solution = vector_from_vec_obj(b_in);
  print_vector_norm(&temp_solution, "Z1");
  NumericVectorDistributed vec_a, vec_b;
  vec_a.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  vec_b.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  
  for(unsigned int i = n_procs_in_sweep - 1; i > 0; i--) {
    S_inv(&temp_solution, &vec_a, i == rank);
    print_vector_norm(&vec_a, "A1");
    vec_b = off_diagonal_product(i, i-1, &vec_a);
    print_vector_norm(&vec_b, "A2");
    subtract_vectors(&temp_solution, &vec_b);
    vec_b.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
    vec_a.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  }
  
  copy_local_part(&temp_solution, &vec_b);
  print_vector_norm(&temp_solution, "B1");
  S_inv(&vec_b, &temp_solution, true);
  print_vector_norm(&temp_solution, "B2");
  vec_a.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  vec_b.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  
  for(unsigned int i = 0; i < n_procs_in_sweep-1; i++) {
    vec_a = off_diagonal_product(i, i+1, &temp_solution);
    print_vector_norm(&vec_a, "C1");
    S_inv(&vec_a, &vec_b, rank == i+1);
    subtract_vectors(&temp_solution, &vec_b);
    print_vector_norm(&temp_solution, "C2");
    vec_b.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  }
  step_counter ++;
  set_x_out_from_u(u_out, &temp_solution);
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

void NonLocalProblem::S_inv(NumericVectorDistributed * src, NumericVectorDistributed * dst, bool execute_locally) {
  dst->reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  if(execute_locally) {
    set_child_rhs_from_vector(src);
    child->solve();
    set_vector_from_child_solution(dst);
  }
  dst->compress(dealii::VectorOperation::insert);
}

void NonLocalProblem::set_vector_from_child_solution(NumericVectorDistributed * in_u) {
  child->solution.extract_subvector_to(vector_copy_child_indeces, vector_copy_array);
  in_u->set(vector_copy_own_indices, vector_copy_array);
}

void NonLocalProblem::set_child_rhs_from_vector(NumericVectorDistributed * in_u) {
  in_u->extract_subvector_to(vector_copy_own_indices, vector_copy_array);
  child->rhs.set(vector_copy_child_indeces, vector_copy_array);
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting");
  child->reinit();
  
  make_constraints();
  // print_diagnosis_data();
  
  make_sparsity_pattern();

  reinit_rhs();
  std::vector<unsigned int> local_rows;
  for(unsigned int p = 0; p < Geometry.levels[level].dof_distribution.size(); p++) {
    local_rows.push_back(Geometry.levels[level].dof_distribution[p].n_elements());
  }

  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  direct_solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  solution_error.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
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

  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  GlobalTimerManager.switch_context("initialize", level);
  child->initialize();
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(GlobalMPI.communicators_by_level[level]);
  rank = dealii::Utilities::MPI::this_mpi_process(GlobalMPI.communicators_by_level[level]);
  initialize_index_sets();
  reinit();
  
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
  write_multifile_output("solution", solution);
  return "";
}

void NonLocalProblem::store_solution(NumericVectorLocal u) {
  stored_solutions.push_back(u);
}

void NonLocalProblem::write_output_for_stored_solution(unsigned int index) {
  NumericVectorLocal local_solution(Geometry.levels[level].inner_domain->n_locally_active_dofs);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    local_solution[i] = stored_solutions[index][i];
  }
  Geometry.levels[level].inner_domain->output_results("solution_nr_" + std::to_string(index), local_solution);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    local_solution[i] = ((std::complex<double>)(stored_solutions[index])[i]) - (std::complex<double>)direct_solution[Geometry.levels[level].inner_first_dof + i];
  }
  Geometry.levels[level].inner_domain->output_results("error_of_solution_nr_" + std::to_string(index), local_solution);
}

void NonLocalProblem::write_multifile_output(const std::string & in_filename, const NumericVectorDistributed field) {
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
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level]);
            delete[] sent_rows;
            delete[] sent_cols;
          }
          // receive part
          if(recv_buffer[other_proc] > 0) {
            // There is something to receive
            unsigned int * received_rows = new unsigned int [recv_buffer[other_proc]];
            unsigned int * received_cols = new unsigned int [recv_buffer[other_proc]];
            MPI_Recv(received_rows, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level], 0);
            MPI_Recv(received_cols, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level], 0);
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
            MPI_Recv(received_rows, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level], 0);
            MPI_Recv(received_cols, recv_buffer[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level], 0);
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
            MPI_Send(sent_rows, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level]);
            MPI_Send(sent_cols, entries_by_proc[other_proc], MPI_UNSIGNED, other_proc, 0, GlobalMPI.communicators_by_level[level]);
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
  ret.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  const unsigned int n_loc_dofs = own_dofs.n_elements();
  ComplexNumber * values = new ComplexNumber[n_loc_dofs];
  VecGetValues(in_v, n_loc_dofs, locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < n_loc_dofs; i++) {
     vector_copy_array[i] = values[i];
  }
  ret.set(vector_copy_own_indices, vector_copy_array);
  ret.compress(dealii::VectorOperation::insert);
  delete[] values;
  return ret;
}

void NonLocalProblem::copy_local_part(NumericVectorDistributed * src, NumericVectorDistributed * dst) {
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
    ComplexNumber temp = src->operator()(own_dofs.nth_index_in_set(i));
    dst->operator[](own_dofs.nth_index_in_set(i)) = temp;
  }
  dst->compress(dealii::VectorOperation::insert);
}

NumericVectorDistributed NonLocalProblem::off_diagonal_product(unsigned int i, unsigned int j, NumericVectorDistributed * in_v) {
  NumericVectorDistributed ret;
  ret.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  NumericVectorDistributed left;
  left.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  if(rank == i) {
    for(unsigned int index = 0; index < own_dofs.n_elements(); index++) {
      ComplexNumber temp = in_v->operator[](own_dofs.nth_index_in_set(index));
      left[own_dofs.nth_index_in_set(index)] = temp;
    }
  } 
  left.compress(VectorOperation::insert);
  if(i < j) {
    matrix->vmult(ret, left);
  } else {
    matrix->Tvmult(ret, left);
  }
  if(rank != j) {
    for(unsigned int index = 0; index < own_dofs.n_elements(); index++) {
      ret[own_dofs.nth_index_in_set(index)] = 0;
    }
  }
  ret.compress(dealii::VectorOperation::insert);
  return ret;
}

void NonLocalProblem::subtract_vectors(NumericVectorDistributed * a, NumericVectorDistributed * b) {
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
    ComplexNumber a_val = a->operator[](own_dofs.nth_index_in_set(i));
    ComplexNumber b_val = b->operator[](own_dofs.nth_index_in_set(i));
    a->operator[](own_dofs.nth_index_in_set(i)) = a_val - b_val;
  }
  a->compress(dealii::VectorOperation::insert);
}

void NonLocalProblem::print_vector_norm(NumericVectorDistributed * in_v, std::string marker) {
  in_v->extract_subvector_to(vector_copy_own_indices, vector_copy_array);
  double local_norm = 0.0;
  for(unsigned int i = 0; i < vector_copy_array.size(); i++) {
    local_norm += std::abs(vector_copy_array[i])*std::abs(vector_copy_array[i]);
  }
  local_norm = dealii::Utilities::MPI::sum(local_norm, MPI_COMM_WORLD);
  if(GlobalParams.MPI_Rank == 0) {
    std::cout << marker << ": " << std::sqrt(local_norm) << std::endl;
  }
}