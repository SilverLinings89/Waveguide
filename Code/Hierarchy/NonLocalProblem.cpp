#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "../Helpers/staticfunctions.h"
#include "HierarchicalProblem.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <mpi.h>
#include <petscsystypes.h>

#include <algorithm>
#include <iterator>
#include <ratio>
#include <string>
#include <vector>

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
  
  for(unsigned int i = 0; i < 6; i++) {
    is_sweeping_hsie_surface[i] = false;
  }

  switch (sweeping_direction) {
    case SweepingDirection::X:
      is_sweeping_hsie_surface[0] = GlobalParams.Index_in_x_direction > 0;
      is_sweeping_hsie_surface[1] = GlobalParams.Index_in_x_direction < GlobalParams.Blocks_in_x_direction - 1;
      break;
    case SweepingDirection::Y:
      is_sweeping_hsie_surface[2] = GlobalParams.Index_in_y_direction > 0;
      is_sweeping_hsie_surface[3] = GlobalParams.Index_in_y_direction < GlobalParams.Blocks_in_y_direction - 1;
      break;
    case SweepingDirection::Z:
      is_sweeping_hsie_surface[4] = GlobalParams.Index_in_z_direction > 0;
      is_sweeping_hsie_surface[5] = GlobalParams.Index_in_z_direction < GlobalParams.Blocks_in_z_direction - 1;
  }

  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  is_mpi_cache_ready = false;
}

void NonLocalProblem::init_solver_and_preconditioner() {
  // dealii::PETScWrappers::PreconditionNone pc_none;
  // pc_none.initialize(*matrix);
  KSPCreate(GlobalMPI.communicators_by_level[level], &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetOperators(ksp, *matrix, *matrix);
  KSPSetType(ksp, KSPGMRES);
  PCSetType(pc,PCSHELL);
  pc_create(&shell, this);
  PCShellSetApply(pc,pc_apply);
  PCShellSetContext(pc, (void*) &shell);
  KSPSetPC(ksp, pc);
}

void NonLocalProblem::reinit_rhs() {
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[level]);
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
  delete[] mpi_cache;
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

void remove_double_entries_from_vector(std::vector<DofNumber> * in_vector) {
  std::sort(in_vector->begin(), in_vector->end());
  std::vector<unsigned int> alternative;
  if(in_vector->size() > 0) alternative.push_back(in_vector->operator[](0));
  for(unsigned int i = 1; i < in_vector->size(); i++) {
    if(! ( in_vector->operator[](i) == in_vector->operator[](i-1))) {
      alternative.push_back(in_vector->operator[](i));
    }
  }
  in_vector->clear();
  for(unsigned int i = 0; i < alternative.size(); i++) {
    in_vector->push_back(alternative[i]);
  }
  if(in_vector->size() == 0) std::cout << "Logic error detected..." << std::endl;
}

void NonLocalProblem::print_diagnosis_data() {
  if(rank == 0) {
    std::cout << "Temp dof data:" << std::endl;
    std::cout << "Pair 1:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(25620);
  print_dof_details(76335);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 2:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(101115);
  print_dof_details(142023);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 3:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(206871);
  print_dof_details(165522);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 4:" << std::endl;
  }
  print_dof_details(141183);
  print_dof_details(99834);
  MPI_Barrier(MPI_COMM_WORLD);
}

void NonLocalProblem::assemble() {
  print_info("NonLocalProblem::assemble", "Begin assembly");
  Geometry.inner_domain->assemble_system(Geometry.levels[level].inner_first_dof, &constraints, matrix, &rhs);
  print_info("NonLocalProblem::assemble", "Inner assembly done. Assembling boundary method contributions.");
  for(unsigned int i = 0; i< 6; i++) {
    if(Geometry.levels[level].is_surface_truncated[i]) {
      Geometry.levels[level].surfaces[i]->fill_matrix(matrix, &rhs, &constraints);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Compress matrix.");
  matrix->compress(dealii::VectorOperation::add);
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Compress vectors.");
  rhs.compress(dealii::VectorOperation::add);
  MPI_Barrier(MPI_COMM_WORLD);
  solution.compress(dealii::VectorOperation::add);
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("NonLocalProblem::assemble", "Assemble child.");
  child->assemble();
  print_info("NonLocalProblem::assemble", "End assembly.");
}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(Geometry.levels[level].n_local_dofs);
  return ret;
}

void NonLocalProblem::solve() {
  
  constraints.set_zero(solution);
  
  KSPSetConvergenceTest(ksp, &convergence_test, reinterpret_cast<void *>(&sc),nullptr);
  KSPSetTolerances(ksp, 0.000001, 1.0, 1000, 30);
  KSPSetUp(ksp);
  PetscErrorCode ierr = KSPSolve(ksp, rhs, solution);
  
  constraints.distribute(solution);
  
  if(ierr != 0) {
    std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    throw new ExcPETScError(ierr);
  }
}

void NonLocalProblem::propagate_up(){
  const unsigned int count = Geometry.inner_domain->n_dofs;
  // Propagate the inner values. HSIE-dofs can be added later.
  for(unsigned int i = 0; i < count; i++) {
    u[i] = child->solution(Geometry.levels[level-1].inner_first_dof + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[level].is_surface_truncated[i]) {
      for(unsigned int j = 0; j < Geometry.levels[level].surfaces[i]->dof_counter; j++) {
        u[Geometry.levels[level].surface_first_dof[i] - Geometry.levels[level].inner_first_dof + j] = child->solution(Geometry.levels[level-1].surface_first_dof[i] + j);
      }
    }
  }
}

void NonLocalProblem::apply_sweep(Vec x_in, Vec x_out) {
  print_info("NonLocalProblem::apply_sweep", "Start");
  MPI_Barrier(MPI_COMM_WORLD);
  /**
   * Algorithm for 4 procs:
   *  u(E_0) = b(E_0)
   *  u(E_1) = b(E_1)
   *  u(E_2) = b(E_2)
   *  u(E_3) = b(E_3)

   *  u(E_2) = u(E_2) - A(E_2, E_3) H_3^{-1} u(E_3)
   *  u(E_1) = u(E_1) - A(E_1, E_2) S_2^{-1} u(E_2)
   *  u(E_0) = u(E_0) - A(E_0, E_1) S_1^{-1} u(E_1)

   *  u(E_3) = H^{-1}_3 u(E_3)
   *  u(E_2) = S^{-1}_2 u(E_2)
   *  u(E_1) = S^{-1}_1 u(E_1)
   *  u(E_0) = S^{-1}_0 u(E_0)

   *  u(E_1) = u(E_1) - S_1^{-1} A(E_1, E_0)u(E_0) 
   *  u(E_2) = u(E_2) - S_2^{-1} A(E_2, E_1)u(E_1)
   *  u(E_3) = u(E_3) - H_3^{-1} A(E_3, E_2)u(E_2)
   **/
  
  // First compy values as in u(E_i) = b(E_i)
  setSolutionFromVector(x_in);

  if(is_highest_in_sweeping_direction()) {
    std::vector<ComplexNumber> rhs_values = UpperBlockProductAfterH();
    send_local_lower_dofs(rhs_values);
  } else {
    receive_local_upper_dofs();
    if(!is_lowest_in_sweeping_direction()) {
      std::vector<ComplexNumber> rhs_values = UpperBlockProductAfterH();
      send_local_lower_dofs(rhs_values);
    }
  }
  
  setChildRhsComponentsFromU(); // sets rhs in child.
  child->solve(); // applies H^{-1} or S^{-1}.
  propagate_up(); // updates u.
  
  if(is_lowest_in_sweeping_direction()) {
    std::vector<ComplexNumber> rhs_values = LowerBlockProduct();
    send_local_upper_dofs(rhs_values);
  } else {
    receive_local_lower_dofs_and_H();
    if(!is_highest_in_sweeping_direction()) {
      std::vector<ComplexNumber> rhs_values = LowerBlockProduct();
      send_local_upper_dofs(rhs_values);
    }
  }
  
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
     values[i] = u[i];
  }
  VecSetValues(x_out, own_dofs.n_elements(), locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(x_out);
  VecAssemblyEnd(x_out);
  MPI_Barrier(MPI_COMM_WORLD);
  delete[] values;
  print_info("NonLocalProblem::apply_sweep", "End");
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting");
  child->reinit();
  
  make_constraints();
  print_diagnosis_data();
  
  make_sparsity_pattern();

  reinit_rhs();
  u = new ComplexNumber[Geometry.levels[level].n_local_dofs];
  for(unsigned int i= 0; i < Geometry.levels[level].n_local_dofs; i++) {
    u[i] = 0;
  }
  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  matrix->reinit(Geometry.levels[level].dof_distribution[rank], Geometry.levels[level].dof_distribution[rank], sp, GlobalMPI.communicators_by_level[level]);
  // matrix->reinit(MPI_COMM_WORLD, Geometry.levels[level].n_total_level_dofs, Geometry.levels[level].n_total_level_dofs, Geometry.levels[level].n_local_dofs, Geometry.levels[level].n_local_dofs, Geometry.inner_domain->dof_handler.max_couplings_between_dofs(), false,  100);
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  child->initialize();
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(GlobalMPI.communicators_by_level[level]);
  rank = dealii::Utilities::MPI::this_mpi_process(GlobalMPI.communicators_by_level[level]);

  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();

  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
}

void NonLocalProblem::initialize_index_sets() {
  lower_sweeping_interface_id = compute_lower_interface_id();
  upper_sweeping_interface_id = compute_upper_interface_id();

  lower_interface_dofs = compute_interface_dof_set(lower_sweeping_interface_id);
  upper_interface_dofs = compute_interface_dof_set(upper_sweeping_interface_id);

  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
  initialize_own_dofs();
}

void NonLocalProblem::initialize_own_dofs() {
  own_dofs = IndexSet(Geometry.levels[level].n_total_level_dofs);
  unsigned int first_dof = Geometry.levels[level].inner_first_dof;
  unsigned int last_dof = Geometry.levels[level].surface_first_dof[5] + Geometry.levels[level].surfaces[5]->dof_counter;
  own_dofs.add_range(first_dof, last_dof);
}

DofCount NonLocalProblem::compute_interface_dofs(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  DofCount ret = 0;
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      ret += Geometry.inner_domain->get_surface_dof_vector_for_boundary_id(interface_id).size();
    } else {
      if(i != opposing_interface_id) {
        if(Geometry.levels[level].is_surface_truncated[i]) {
          ret += Geometry.levels[level].surfaces[i]->get_dof_association_by_boundary_id(i).size();
        }
      }
    }
  }
  return ret;
}

dealii::IndexSet NonLocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  dealii::IndexSet ret(Geometry.levels[level].n_total_level_dofs);
  std::vector<InterfaceDofData> current = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(interface_id, level);
  for(unsigned int j = 0; j < current.size(); j++) {
    ret.add_index(current[j].index);
  }
  
  for(unsigned int i = 0; i < 6; i++) {
    if( i != interface_id && !are_opposing_sites(i,interface_id)) {
      if(Geometry.levels[level].is_surface_truncated[i]) {
        std::vector<InterfaceDofData> current = Geometry.levels[level].surfaces[i]->get_dof_association_by_boundary_id(interface_id);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index); 
        }
      }
    }
  }
  return ret;
}

auto NonLocalProblem::compute_lower_interface_id() -> BoundaryId {
  if (this->sweeping_direction == SweepingDirection::X) {
    return 0;
  }
  if (this->sweeping_direction == SweepingDirection::Y) {
    return 2;
  }
  if (this->sweeping_direction == SweepingDirection::Z) {
    return 4;
  }
  return 0;
}

auto NonLocalProblem::compute_upper_interface_id() -> BoundaryId {
  if (this->sweeping_direction == SweepingDirection::X) {
    return 1;
  }
  if (this->sweeping_direction == SweepingDirection::Y) {
    return 3;
  }
  if (this->sweeping_direction == SweepingDirection::Z) {
    return 5;
  }
  return 0;
}

auto NonLocalProblem::compute_lower_interface_dof_count() -> DofCount {
  return compute_interface_dofs(compute_lower_interface_id());
}

auto NonLocalProblem::compute_upper_interface_dof_count() -> DofCount {
  return compute_interface_dofs(compute_upper_interface_id());
}

auto NonLocalProblem::get_center() -> Position const {
  Position local_contribution = Geometry.get_local_center();
  double x = dealii::Utilities::MPI::min_max_avg(local_contribution[0], GlobalMPI.communicators_by_level[level]).avg;
  double y = dealii::Utilities::MPI::min_max_avg(local_contribution[1], GlobalMPI.communicators_by_level[level]).avg;
  double z = dealii::Utilities::MPI::min_max_avg(local_contribution[2], GlobalMPI.communicators_by_level[level]).avg;
  return {x,y,z};
}

void NonLocalProblem::H_inverse() {
  child->solve();
}

bool NonLocalProblem::is_lowest_in_sweeping_direction() {
  if(sweeping_direction == SweepingDirection::X) {
    if(GlobalParams.Index_in_x_direction == 0) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Y) {
    if(GlobalParams.Index_in_y_direction == 0) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Z) {
    if(GlobalParams.Index_in_z_direction == 0) {
      return true;
    }
  }
  return false;
}

bool NonLocalProblem::is_highest_in_sweeping_direction() {
  if(sweeping_direction == SweepingDirection::X) {
    if(GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction-1) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Y) {
    if(GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction-1) {
      return true;
    }
  }
  if(sweeping_direction == SweepingDirection::Z) {
    if(GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction-1) {
      return true;
    }
  }
  return false;
}

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

void NonLocalProblem::reinit_mpi_cache(unsigned int n_elements) {
  if(is_mpi_cache_ready) {
    delete[] mpi_cache;
  }
  mpi_cache = new ComplexNumber[n_elements];
  for(unsigned int i = 0; i < n_elements; i++) {
    mpi_cache[i] = 0;
  }
  is_mpi_cache_ready = true;
}

void NonLocalProblem::send_local_lower_dofs(std::vector<ComplexNumber> values) {
  reinit_mpi_cache(values.size());
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    mpi_cache[i] = values[i];
  }
  MPI_Send(&mpi_cache[0], values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::receive_local_lower_dofs_and_H() {
  const unsigned int n_elements = lower_interface_dofs.n_elements();
  reinit_mpi_cache(n_elements);
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], n_elements, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  child->reinit_rhs();
  const unsigned int count = Geometry.inner_domain->n_dofs;
  for(unsigned int i = 0; i < n_elements; i++) {
    const DofNumber dof = lower_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof + Geometry.levels[level-1].inner_first_dof;
    child->rhs[dof] = mpi_cache[i];
  }
  child->rhs.compress(VectorOperation::insert);
  std::cout << "After receiving rhs has norm " << child->rhs.l2_norm() << std::endl;
  child->solve();
  std::cout << "After solve rhs has norm " << child->solution.l2_norm() << std::endl;
  for(unsigned int i = 0; i < count; i++) {
    u[i] -= (ComplexNumber)child->solution(Geometry.levels[level-1].inner_first_dof + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[level].is_surface_truncated[i]) {
      for(unsigned int j = 0; j < Geometry.levels[level].surfaces[i]->dof_counter; j++) {
        u[Geometry.levels[level].surface_first_dof[i] - Geometry.levels[level].inner_first_dof + j] -= ComplexNumber(child->solution(Geometry.levels[level-1].surface_first_dof[i] + j).real(), child->solution(Geometry.levels[level-1].surface_first_dof[i] + j).imag());
      }
    }
  }
}

void NonLocalProblem::receive_local_upper_dofs() {
  reinit_mpi_cache(upper_interface_dofs.n_elements());
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], upper_interface_dofs.n_elements(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    u[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof] -= mpi_cache[i];
  }
}

void NonLocalProblem::send_local_upper_dofs(std::vector<ComplexNumber> values) {
  reinit_mpi_cache(values.size());
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  for(unsigned int i = 0; i < values.size(); i++) {
    mpi_cache[i] = values[i];
  }
  MPI_Send(&mpi_cache[0], values.size(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::update_mismatch_vector(BoundaryId) {
  // rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[level]);
  NumericVectorDistributed temp_solution(child->own_dofs, GlobalMPI.communicators_by_level[child->level]);
  NumericVectorDistributed temp_rhs(child->own_dofs, GlobalMPI.communicators_by_level[child->level]);
  child->rhs_mismatch = 0;
  // Copy dof values for inner problem
  for(unsigned int index = 0; index < Geometry.inner_domain->n_dofs; index++) {
    temp_solution[Geometry.levels[level-1].inner_first_dof + index] = solution[Geometry.levels[level].inner_first_dof + index];
  }
  // Copy dof values for any HSIE dofs of non-sweeping interfaces.
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(surface != get_lower_boundary_id_for_sweeping_direction(sweeping_direction) && surface != get_upper_boundary_id_for_sweeping_direction(sweeping_direction)) {
      if(Geometry.levels[level].is_surface_truncated[surface]) {
        for(unsigned int index = 0; index < Geometry.levels[level].surfaces[surface]->dof_counter; index++ ){
          temp_solution[Geometry.levels[level-1].surface_first_dof[surface] + index] = solution[Geometry.levels[level].surface_first_dof[surface] + index];
        }
      }
    }
  }
  // Dof values of sweeping interfaces are left at zero, leading to a 
  // non-zero solution (since we solved this problem exacty before with 
  // the boundary elements for the sweeping interface, we now get a non-zero rhs).
  child->matrix->vmult(temp_rhs, temp_solution);
  for(unsigned int index = 0; index < Geometry.inner_domain->n_dofs; index++) {
    rhs_mismatch[Geometry.levels[level].inner_first_dof + index] = temp_rhs[Geometry.levels[level-1].inner_first_dof + index];
  }
}

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {
  BoundaryId bid = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  NumericVectorLocal ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = rhs_mismatch[Geometry.levels[level].inner_first_dof + is.nth_index_in_set(i)];
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::extract_local_lower_dofs() {
  BoundaryId bid = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  NumericVectorLocal ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = rhs_mismatch[Geometry.levels[level].inner_first_dof + is.nth_index_in_set(i)];
  }
  return ret;
}

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
}

void NonLocalProblem::output_results() {
  /**
  for(unsigned int dof = 0; dof < Geometry.inner_domain->n_dofs; dof++) {
    get_local_problem()->solution[dof] = solution[Geometry.levels[level].inner_first_dof + dof];
  }
  get_local_problem()->output_results();
  **/ 
}

std::vector<ComplexNumber> NonLocalProblem::UpperBlockProductAfterH() {
  IndexSet is = lower_interface_dofs;
  setChildRhsComponentsFromU();
  child->solve();
  child->update_mismatch_vector(compute_upper_interface_id());
  
  std::vector<ComplexNumber> ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[is.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof];
  }
  std::cout << "In rank " << rank << " Upper Block Product Norm is " << l2_norm_of_vector(ret) << std::endl;
  return ret;
}

std::vector<ComplexNumber> NonLocalProblem::LowerBlockProduct() {
  setChildSolutionComponentsFromU();
  child->update_mismatch_vector(compute_upper_interface_id());
  std::vector<ComplexNumber> ret(upper_interface_dofs.n_elements());
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[upper_interface_dofs.nth_index_in_set(i) - Geometry.levels[level].inner_first_dof + Geometry.levels[level-1].inner_first_dof];
  }
  std::cout << "In rank " << rank << " Lower Block Product Norm is " << l2_norm_of_vector(ret) << " and rhs_mismatch norm " << child->rhs_mismatch.l2_norm() << std::endl;
  return ret;
}

void NonLocalProblem::setSolutionFromVector(Vec x_in) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  VecGetValues(x_in, own_dofs.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < Geometry.levels[level].n_local_dofs; i++) {
    u[i] = values[i]; 
  }
  delete[] values;
}

void NonLocalProblem::setChildSolutionComponentsFromU() {
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    child->solution[Geometry.levels[level-1].inner_first_dof + i] = u[i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(Geometry.levels[level].is_surface_truncated[surface]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
        child->solution[Geometry.levels[level-1].surface_first_dof[surface] + i] = u[Geometry.levels[level].surface_first_dof[surface] - Geometry.levels[level].inner_first_dof + i];
      }
    } else {
      if(Geometry.levels[level-1].is_surface_truncated[surface]) {
        std::vector<InterfaceDofData> vec = Geometry.levels[level].surfaces[surface]->get_dof_association();
        std::sort(vec.begin(), vec.end(), compareDofDataByGlobalIndex);
        unsigned int index = 0;
        for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
          while(vec[index].index < i){
            index++;
          }
          if(i != vec[index].index){
            child->solution[Geometry.levels[level-1].surface_first_dof[surface] + i] = 0;
          } 
        }
      }
    }
  }
  child->solution.compress(VectorOperation::insert);
}

void NonLocalProblem::setChildRhsComponentsFromU() {
  for(unsigned int i = 0; i < Geometry.inner_domain->n_dofs; i++) {
    child->rhs[Geometry.levels[level-1].inner_first_dof + i] = u[i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(Geometry.levels[level].is_surface_truncated[surface]) {
      for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
        child->rhs[Geometry.levels[level-1].surface_first_dof[surface] + i] = u[Geometry.levels[level].surface_first_dof[surface] - Geometry.levels[level].inner_first_dof + i];
      }
    } else {
      if(Geometry.levels[level-1].is_surface_truncated[surface]) {
        for(unsigned int i = 0; i < Geometry.levels[level].surfaces[surface]->dof_counter; i++) {
          // child->rhs[Geometry.levels[level-1].surface_first_dof[surface] + i] = 0;
        }
      }
    }
  }
  child->rhs.compress(VectorOperation::insert);
}

DofOwner NonLocalProblem::get_dof_owner(unsigned int dof) {
  DofOwner ret;
  if(dof < Geometry.levels[level].inner_first_dof || dof > Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs) {
    std::cout << "get_dof_data was called for a dof that is not locally owned" << std::endl;
    return ret;
  }
  ret.owner = rank;
  ret.is_boundary_dof = dof > Geometry.levels[level].surface_first_dof[0];
  if(ret.is_boundary_dof) {
    for(unsigned int i = 0; i < 5; i++) {
      if(dof < Geometry.levels[level].surface_first_dof[i+1]) {
        ret.surface_id = i;
        return ret;
      }
    }
  }
  ret.surface_id = 5;
  return ret;
}

void NonLocalProblem::print_dof_details(unsigned int dof) {
  if(is_dof_locally_owned(dof)) {
    DofOwner data = get_dof_owner(dof);
    if(!data.is_boundary_dof) {
      std::cout << "Dof " + std::to_string(dof) + " is a inner dof on process " + std::to_string(rank) << std::endl;
    } else {
      std::cout << "Dof " + std::to_string(dof) + " is a boundary dof on process " + std::to_string(rank) + " for boundary " + std::to_string(data.surface_id)<< std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

bool NonLocalProblem::is_dof_locally_owned(unsigned int dof) {
  return (dof >= Geometry.levels[level].inner_first_dof && dof < Geometry.levels[level].inner_first_dof + Geometry.levels[level].n_local_dofs);
}
