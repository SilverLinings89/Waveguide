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

NonLocalProblem::NonLocalProblem(unsigned int local_level) :
  HierarchicalProblem(local_level, static_cast<SweepingDirection> (2 + GlobalParams.HSIE_SWEEPING_LEVEL - local_level)),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true)
{
  if(local_level > 1) {
    child = new NonLocalProblem(local_level - 1);
  } else {
    child = new LocalProblem();
  }
  
  for(unsigned int i = 0; i < 6; i++) {
    is_hsie_surface[i] = child->is_hsie_surface[i];
    is_sweeping_hsie_surface[i] = false;
  }

  switch (sweeping_direction) {
    case SweepingDirection::X:
      is_hsie_surface[0] = GlobalParams.Index_in_x_direction == 0;
      is_hsie_surface[1] = GlobalParams.Index_in_x_direction == GlobalParams.Blocks_in_x_direction - 1;
      is_sweeping_hsie_surface[0] = GlobalParams.Index_in_x_direction > 0;
      is_sweeping_hsie_surface[1] = GlobalParams.Index_in_x_direction < GlobalParams.Blocks_in_x_direction - 1;
      break;
    case SweepingDirection::Y:
      is_hsie_surface[2] = GlobalParams.Index_in_y_direction == 0;
      is_hsie_surface[3] = GlobalParams.Index_in_y_direction == GlobalParams.Blocks_in_y_direction - 1;
      is_sweeping_hsie_surface[2] = GlobalParams.Index_in_y_direction > 0;
      is_sweeping_hsie_surface[3] = GlobalParams.Index_in_y_direction < GlobalParams.Blocks_in_y_direction - 1;
      break;
    case SweepingDirection::Z:
      is_hsie_surface[4] = (GlobalParams.Index_in_z_direction == 0 ) && child->is_hsie_surface[4];
      is_hsie_surface[5] = GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1;
      is_sweeping_hsie_surface[4] = GlobalParams.Index_in_z_direction > 0;
      is_sweeping_hsie_surface[5] = GlobalParams.Index_in_z_direction < GlobalParams.Blocks_in_z_direction - 1;
  }

  n_own_dofs = 0;
  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  is_mpi_cache_ready = false;
}

void NonLocalProblem::init_solver_and_preconditioner() {
  // dealii::PETScWrappers::PreconditionNone pc_none;
  // pc_none.initialize(*matrix);
  KSPCreate(GlobalMPI.communicators_by_level[local_level], &ksp);
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
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[local_level]);
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
  delete[] mpi_cache;
}

void NonLocalProblem::make_constraints_for_local_surface(unsigned int surface) {
  if(is_hsie_surface[surface]) {
    std::vector<InterfaceDofData> from_surface = get_local_problem()->surfaces[surface]->get_dof_association();
    std::vector<InterfaceDofData> from_inner_problem = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
    shift_interface_dof_data(&from_surface, surface_first_dofs[surface]);
    shift_interface_dof_data(&from_inner_problem, first_own_index);
    AffineConstraints<ComplexNumber> new_constraints = get_affine_constraints_for_InterfaceData(from_surface, from_inner_problem, total_number_of_dofs_on_level);
    constraints.merge(new_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
  }
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

void NonLocalProblem::make_constraints_for_non_local_surface(unsigned int surface) {
  std::vector<InterfaceDofData> from_inner_problem = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
  shift_interface_dof_data(&from_inner_problem, first_own_index);
  for(unsigned int i = 0; i < 6; i++) {
    if((i!= surface && !are_opposing_sites(surface, i)) && is_hsie_surface[i]) {
      std::vector<InterfaceDofData> surf_data = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(surface);
      shift_interface_dof_data(&surf_data, surface_first_dofs[i]);
      from_inner_problem.insert(from_inner_problem.end(), surf_data.begin(), surf_data.end());
    }
  }
  std::vector<std::pair<unsigned int, unsigned int>> old_constraints;
  for(unsigned int i = 0; i < from_inner_problem.size(); i++) {
    if(constraints.is_constrained(from_inner_problem[i].index)) {
      const std::vector<std::pair<unsigned int, ComplexNumber>> * cons = constraints.get_constraint_entries(from_inner_problem[i].index);
      if(cons->size() != 1) {
        std::cout << "Weird case" << std::endl;
      } else {
        old_constraints.emplace_back(from_inner_problem[i].index, (*cons)[0].first);
      }
    }
  }

  unsigned int n_dofs_on_surface = from_inner_problem.size();
  std::pair<bool, unsigned int> partner_data = GlobalMPI.get_neighbor_for_interface(get_direction_for_boundary_id(surface));
  if(!partner_data.first) {
    std::cout << "There was an error finding the partner process in NonlocalProblem::make_constraints_for_non_local_surface" << std::endl;
  }
  unsigned int partner_index = partner_data.second;
  double * x_values = new double[n_dofs_on_surface];
  double * y_values = new double[n_dofs_on_surface];
  double * z_values = new double[n_dofs_on_surface];
  unsigned int * orders = new unsigned int[n_dofs_on_surface];
  DofNumber * indices = new unsigned int[n_dofs_on_surface];
  for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
    x_values[i] = from_inner_problem[i].base_point[0];
    y_values[i] = from_inner_problem[i].base_point[1];
    z_values[i] = from_inner_problem[i].base_point[2];
    indices[i] = from_inner_problem[i].index;
    coupling_dofs[surface].emplace_back(indices[i], 0);
  }
  
  MPI_Sendrecv_replace(x_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  MPI_Sendrecv_replace(y_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  MPI_Sendrecv_replace(z_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  MPI_Sendrecv_replace(indices, n_dofs_on_surface, MPI_UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  MPI_Sendrecv_replace(orders, n_dofs_on_surface, MPI_UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  std::vector<InterfaceDofData> from_other_process;
  for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
    coupling_dofs[surface][i].second = indices[i];
    InterfaceDofData item;
    item.base_point[0] = x_values[i];
    item.base_point[1] = y_values[i];
    item.base_point[2] = z_values[i];
    item.order = orders[i];
    item.index = indices[i];
    from_other_process.push_back(item);
  }
  AffineConstraints<ComplexNumber> temp_constraints = get_affine_constraints_for_InterfaceData(from_inner_problem, from_other_process, total_number_of_dofs_on_level);
  constraints.merge(temp_constraints, AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
  
  dealii::IndexSet is(total_number_of_dofs_on_level);
  is.add_range(0, total_number_of_dofs_on_level);
  AffineConstraints<ComplexNumber> neighbor_boundary_to_inner_constraints(is);
  for(unsigned int i = 0; i < from_inner_problem.size(); i++) {
    for(unsigned int j = 0; j < old_constraints.size(); j++) {
      if(from_inner_problem[i].index == old_constraints[j].first) {
        neighbor_boundary_to_inner_constraints.add_line(old_constraints[j].second);
        neighbor_boundary_to_inner_constraints.add_entry(old_constraints[j].second, from_other_process[i].index, 1.0);
      }
      if(from_inner_problem[i].index == old_constraints[j].second) {
        neighbor_boundary_to_inner_constraints.add_line(old_constraints[j].first);
        neighbor_boundary_to_inner_constraints.add_entry(old_constraints[j].first, from_other_process[i].index, 1.0);
      }
    }
  }
  constraints.merge(neighbor_boundary_to_inner_constraints, AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
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

void NonLocalProblem::make_sparsity_pattern_for_surface(unsigned int surface, DynamicSparsityPattern * dsp) {
  if(is_sweeping_hsie_surface[surface]) {
    for(unsigned int surf = 0; surf < 6; surf++) {
      if(surf != surface && (!are_opposing_sites(surface, surf))) {
        std::pair<bool, unsigned int> partner_data = GlobalMPI.get_neighbor_for_interface(get_direction_for_boundary_id(surface));
        get_local_problem()->surfaces[surf]->fill_sparsity_pattern_for_neighbor(surface, surface_first_dofs[surface], partner_data.second, &constraints, dsp);
      }
    }
  }
  if(GlobalParams.Index_in_z_direction == 0 && surface == 4) {
    // do nothing currently
  } else {
    std::pair<bool, unsigned int> partner_data = GlobalMPI.get_neighbor_for_interface(get_direction_for_boundary_id(surface));
    if(!partner_data.first) {
      std::cout << "There was an error finding the partner process in NonlocalProblem::make_constraints_for_non_hsie_surface" << std::endl;
    }
    unsigned int partner_index = partner_data.second;
    std::vector<std::vector<DofNumber>> couplings(coupling_dofs[surface].size());
    auto it = get_local_problem()->base_problem.dof_handler.begin();
    std::vector<DofNumber> local_dofs(get_local_problem()->base_problem.fe.dofs_per_cell);
    for( ; it != get_local_problem()->base_problem.dof_handler.end(); it++) {
      if(it->at_boundary(surface)) {
        it->get_dof_indices(local_dofs);
        for(unsigned int i = 0; i< local_dofs.size(); i++) local_dofs[i] += first_own_index;
        for(unsigned int local_dof_index = 0; local_dof_index < local_dofs.size(); local_dof_index++) {
          for(unsigned int index_in_coupling = 0; index_in_coupling < coupling_dofs[surface].size(); index_in_coupling++) {
            if(coupling_dofs[surface][index_in_coupling].first == local_dofs[local_dof_index]) {
              for(unsigned int j = 0; j < local_dofs.size(); j++) {
                couplings[index_in_coupling].push_back(local_dofs[j]);
              }
            }
          }
        }
      }
    }
    for(auto it : couplings) {
      remove_double_entries_from_vector(&it);
    }
    unsigned long max_n_couplings = 0;
    for(unsigned int i = 0; i < couplings.size(); i++) {
      if(max_n_couplings < couplings[i].size()) {
        max_n_couplings = couplings[i].size();
      }
    }
    unsigned long send_temp = max_n_couplings;
    MPI_Sendrecv_replace(&send_temp, 1, MPI::UNSIGNED_LONG, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    max_n_couplings = std::max(send_temp, max_n_couplings);
    int* cache = new int[max_n_couplings + 1];
    // for each coupling dof
    for(unsigned int i = 0; i < coupling_dofs[surface].size(); i++) {
      cache[0] = coupling_dofs[surface][i].first;
      // for each dof it is coupled to locally
      for(unsigned int j = 1; j < couplings[i].size()+1; j++) {
        // store the coupling partner in the cache.
        cache[j] = couplings[i][j-1];
      }
      // Fill the array with -1
      for(unsigned int j = couplings[i].size()+1; j < max_n_couplings+1; j++) {
        cache[j] = -1;
      }
      
      MPI_Sendrecv_replace(cache, max_n_couplings + 1, MPI::INT, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
      dsp->add(coupling_dofs[surface][i].first, cache[0]);
      dsp->add(cache[0], coupling_dofs[surface][i].first);
      for(unsigned int j = 1; j < max_n_couplings+1; j++) {
        if(cache[j] != -1) {
          for(unsigned int local_dof_index = 0; local_dof_index < couplings[i].size(); local_dof_index++) {
            dsp->add(couplings[i][local_dof_index], cache[j]);
            dsp->add(cache[j], couplings[i][local_dof_index]);
          }
        }
      }
    }
    fill_dsp_over_mpi(surface, dsp);
  }
}

void NonLocalProblem::make_constraints() {
  print_info("NonLocalProblem::make_constraints", "Start");
  IndexSet total_dofs_global(total_number_of_dofs_on_level);
  total_dofs_global.add_range(0,total_number_of_dofs_on_level);
  constraints.reinit(total_dofs_global);
  
  for(unsigned int i = 0; i < 6; i++) { 
    coupling_dofs[i].clear();
  }

  for (unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]) get_local_problem()->surfaces[i]->setup_neighbor_couplings(is_hsie_surface);
  }
  // couple surface dofs with inner ones.
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_sweeping_hsie_surface[surface]) {
      make_constraints_for_non_local_surface(surface);
    } else {
      make_constraints_for_local_surface(surface);
    }
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 1: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  
  // Edge2Edge Dofs.
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      if ( is_hsie_surface[i] && is_hsie_surface[j]) {
        if (!are_opposing_sites(i, j)) {
          std::vector<InterfaceDofData> lower_face_dofs = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(j);
          std::vector<InterfaceDofData> upper_face_dofs = get_local_problem()->surfaces[j]->get_dof_association_by_boundary_id(i);
          if (lower_face_dofs.size() != upper_face_dofs.size()) {
            std::cout << "ERROR: There was a edge dof count error!" << std::endl
                << "Surface " << i << " offers " << lower_face_dofs.size()
                << " dofs, " << j << " offers " << upper_face_dofs.size() << "."
                << std::endl;
          }
          shift_interface_dof_data(&lower_face_dofs, surface_first_dofs[i]);
          shift_interface_dof_data(&upper_face_dofs, surface_first_dofs[j]);
          AffineConstraints<ComplexNumber> surf_to_surf = get_affine_constraints_for_InterfaceData(lower_face_dofs, upper_face_dofs, total_number_of_dofs_on_level);
          constraints.merge(surf_to_surf, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
        }
      }
    }
  }
  if(!validate_constraints_range()) {
    std::cout << "rank " << rank << " has incorrect constraints entries at b" << std::endl;
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 2: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  // From inner problem
  get_local_problem()->base_problem.make_constraints(&constraints, first_own_index, own_dofs);
  print_info("LocalProblem::make_constraints", "Constraints after phase 3: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  for(unsigned int i = 0; i < 6; i++) { 
    coupling_dofs[i].clear();
  }
  for (unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]) get_local_problem()->surfaces[i]->reset_neighbor_couplings(is_hsie_surface);
  }
  constraints.close();
  // print_diagnosis_data();
  print_info("NonLocalProblem::make_constraints", "End");
}

void NonLocalProblem::print_diagnosis_data() {
  if(rank == 0) {
    std::cout << "Temp dof data:" << std::endl;
    std::cout << "Pair 1:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(3410);
  print_dof_details(5942);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 2:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(13178);
  print_dof_details(9665);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 3:" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print_dof_details(26356);
  print_dof_details(16247);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Pair 4:" << std::endl;
  }
  print_dof_details(39534);
  print_dof_details(36025);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Temp dof data end" << std::endl;
}

void NonLocalProblem::assemble() {
  print_info("NonLocalProblem::assemble", "Begin assembly.");
  get_local_problem()->base_problem.assemble_system(first_own_index, &constraints, matrix, &rhs);
  print_info("NonLocalProblem::assemble", "Inner assembly done. Assembling boundary method contributions.");
  for(unsigned int i = 0; i< 6; i++) {
    if(is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_matrix(matrix, &rhs, surface_first_dofs[i], is_hsie_surface, &constraints);
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
  dealii::Vector<ComplexNumber> ret(n_own_dofs);
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
  const unsigned int count = get_local_problem()->base_problem.n_dofs;
  // Propagate the inner values. HSIE-dofs can be added later.
  for(unsigned int i = 0; i < count; i++) {
    u[i] = child->solution(child->first_own_index + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(child->is_hsie_surface[i] && is_hsie_surface[i]) {
      for(unsigned int j = 0; j < get_local_problem()->surfaces[i]->dof_counter; j++) {
        u[surface_first_dofs[i] - first_own_index + j] = child->solution(child->surface_first_dofs[i] + j);
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
  make_constraints();
  
  // generate_sparsity_pattern();

  reinit_rhs();
  u = new ComplexNumber[n_own_dofs];
  for(unsigned int i= 0; i < n_own_dofs; i++) {
    u[i] = 0;
  }
  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  std::vector<DofCount> n_dofs_by_proc; 
  for(unsigned int i = 0; i < n_procs_in_sweep; i++) {
    n_dofs_by_proc.push_back(index_sets_per_process[i].n_elements());
  }  
  // matrix->reinit(GlobalMPI.communicators_by_level[local_level], sp, n_dofs_by_proc, n_dofs_by_proc, rank, false);
  matrix->reinit(MPI_COMM_WORLD, total_number_of_dofs_on_level, total_number_of_dofs_on_level, n_own_dofs, n_own_dofs, get_local_problem()->base_problem.dof_handler.max_couplings_between_dofs(), false,  100);
  child->reinit();
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  child->initialize();
  n_own_dofs = compute_own_dofs();
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(GlobalMPI.communicators_by_level[local_level]);
  rank = dealii::Utilities::MPI::this_mpi_process(GlobalMPI.communicators_by_level[local_level]);
  index_sets_per_process = dealii::Utilities::MPI::create_ascending_partitioning(GlobalMPI.communicators_by_level[local_level], n_own_dofs);
  own_dofs = index_sets_per_process[rank];
  first_own_index = own_dofs.nth_index_in_set(0);
  for(unsigned int i = 0; i < 6; i++) {
    if(get_local_problem()->is_hsie_surface[i]) {
      surface_dof_associations[i] = get_local_problem()->surfaces[i]->get_dof_association();
    }
  }
  
  surface_first_dofs.clear();
  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();
  unsigned int current = first_own_index + get_local_problem()->base_problem.n_dofs;
  for(unsigned int surface = 0; surface < 6; surface++) {
    surface_first_dofs.push_back(current);
    if(is_hsie_surface[surface] && get_local_problem()->is_hsie_surface[surface]) {
      current += get_local_problem()->surfaces[surface]->dof_counter;
    }
  }
  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
}

void NonLocalProblem::generate_sparsity_pattern() {
  
  dealii::DynamicSparsityPattern dsp = {total_number_of_dofs_on_level, total_number_of_dofs_on_level};
  dealii::IndexSet is(total_number_of_dofs_on_level);
  is.add_range(0, total_number_of_dofs_on_level);
  
  get_local_problem()->base_problem.make_sparsity_pattern(&dsp, first_own_index, &constraints);
  
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface] && get_local_problem()->is_hsie_surface[surface]) {
      get_local_problem()->surfaces[surface]->fill_sparsity_pattern(&dsp, surface_first_dofs[surface], &constraints);
    } else {
      make_sparsity_pattern_for_surface(surface, & dsp);
    }
  }

  sp.copy_from(dsp);
  sp.compress();
}

void NonLocalProblem::initialize_index_sets() {
  lower_sweeping_interface_id = compute_lower_interface_id();
  upper_sweeping_interface_id = compute_upper_interface_id();
  total_number_of_dofs_on_level = 0;
  for (unsigned int i = 0; i < n_procs_in_sweep; i++) {
    total_number_of_dofs_on_level += index_sets_per_process[i].n_elements();
  }
  
  if (rank > 0) {
    dofs_process_below = index_sets_per_process[rank - 1].n_elements();
  }
  if (rank + 1 < n_procs_in_sweep) {
    dofs_process_above = index_sets_per_process[rank + 1].n_elements();
  }
  for(unsigned int i = 0; i < 6; i++) {
    surface_index_sets[i] = compute_interface_dof_set(i);
  }
  
  lower_interface_dofs = compute_interface_dof_set(lower_sweeping_interface_id);
  upper_interface_dofs = compute_interface_dof_set(upper_sweeping_interface_id);
  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}

void NonLocalProblem::initialize_own_dofs() {
  
}

DofCount NonLocalProblem::compute_interface_dofs(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  DofCount ret = 0;
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      ret += get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(interface_id).size();
    } else {
      if(i != opposing_interface_id) {
        if(is_hsie_surface[i] && get_local_problem()->is_hsie_surface[i]) {
          ret += get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(i).size();
        }
      }
    }
  }
  return ret;
}

dealii::IndexSet NonLocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  dealii::IndexSet ret(total_number_of_dofs_on_level);
  std::vector<InterfaceDofData> current = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(interface_id);
  MPI_Barrier(MPI_COMM_WORLD);
  for(unsigned int j = 0; j < current.size(); j++) {
    ret.add_index(current[j].index + first_own_index);
  }
  
  for(unsigned int i = 0; i < 6; i++) {
    if( i != interface_id && !are_opposing_sites(i,interface_id)) {
      if(is_hsie_surface[i] && get_local_problem()->is_hsie_surface[i]) {
        std::vector<InterfaceDofData> current = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(interface_id);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index + surface_first_dofs[i]); 
        }
      }
    }
  }
  return ret;
}

unsigned int NonLocalProblem::compute_own_dofs() {
  DofCount ret = get_local_problem()->base_problem.dof_handler.n_dofs();
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i] && get_local_problem()->is_hsie_surface[i]) {
      ret += get_local_problem()->surfaces[i]->dof_counter;
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
  Position local_contribution = (this->get_local_problem())->get_center();
  double x = dealii::Utilities::MPI::min_max_avg(local_contribution[0],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
  double y = dealii::Utilities::MPI::min_max_avg(local_contribution[1],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
  double z = dealii::Utilities::MPI::min_max_avg(local_contribution[2],
      GlobalMPI.communicators_by_level[this->local_level]).avg;
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
  const unsigned int count = get_local_problem()->base_problem.n_dofs;
  for(unsigned int i = 0; i < n_elements; i++) {
    const DofNumber dof = lower_interface_dofs.nth_index_in_set(i) - first_own_index + child->first_own_index;
    child->rhs[dof] = mpi_cache[i];
  }
  child->rhs.compress(VectorOperation::insert);
  std::cout << "After receiving rhs has norm " << child->rhs.l2_norm() << std::endl;
  child->solve();
  std::cout << "After solve rhs has norm " << child->solution.l2_norm() << std::endl;
  for(unsigned int i = 0; i < count; i++) {
    u[i] -= (ComplexNumber)child->solution(child->first_own_index + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(child->is_hsie_surface[i] && is_hsie_surface[i]) {
      for(unsigned int j = 0; j < get_local_problem()->surfaces[i]->dof_counter; j++) {
        u[surface_first_dofs[i] - first_own_index + j] -= (ComplexNumber)child->solution(child->surface_first_dofs[i] + j);
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
    u[upper_interface_dofs.nth_index_in_set(i) - first_own_index] -= mpi_cache[i];
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
  // rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  NumericVectorDistributed temp_solution(child->own_dofs, GlobalMPI.communicators_by_level[child->local_level]);
  NumericVectorDistributed temp_rhs(child->own_dofs, GlobalMPI.communicators_by_level[child->local_level]);
  child->rhs_mismatch = 0;
  // Copy dof values for inner problem
  for(unsigned int index = 0; index < get_local_problem()->base_problem.n_dofs; index++) {
    temp_solution[child->first_own_index + index] = solution[first_own_index + index];
  }
  // Copy dof values for any HSIE dofs of non-sweeping interfaces.
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(surface != get_lower_boundary_id_for_sweeping_direction(sweeping_direction) && surface != get_upper_boundary_id_for_sweeping_direction(sweeping_direction)) {
      if(child->is_hsie_surface[surface] && is_hsie_surface[surface]) {
        for(unsigned int index = 0; index < get_local_problem()->surfaces[surface]->dof_counter; index++ ){
          temp_solution[child->surface_first_dofs[surface] + index] = solution[surface_first_dofs[surface] + index];
        }
      }
    }
  }
  // Dof values of sweeping interfaces are left at zero, leading to a 
  // non-zero solution (since we solved this problem exacty before with 
  // the boundary elements for the sweeping interface, we now get a non-zero rhs).
  get_local_problem()->matrix->vmult(temp_rhs, temp_solution);
  for(unsigned int index = 0; index < get_local_problem()->base_problem.n_dofs; index++) {
    rhs_mismatch[first_own_index + index] = temp_rhs[child->first_own_index + index];
  }
}

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {
  BoundaryId bid = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  NumericVectorLocal ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = rhs_mismatch[first_own_index + is.nth_index_in_set(i)];
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::extract_local_lower_dofs() {
  BoundaryId bid = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = surface_index_sets[bid];
  NumericVectorLocal ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = rhs_mismatch[first_own_index + is.nth_index_in_set(i)];
  }
  return ret;
}

void NonLocalProblem::compute_solver_factorization() {
  child->compute_solver_factorization();
  // child->output_results();
}

void NonLocalProblem::output_results() {
  for(unsigned int dof = 0; dof < get_local_problem()->base_problem.n_dofs; dof++) {
    get_local_problem()->solution[dof] = solution[first_own_index + dof];
  }
  get_local_problem()->output_results();
}

void NonLocalProblem::fill_dsp_over_mpi(BoundaryId surface, dealii::DynamicSparsityPattern * in_dsp) {
  // fill a local dsp object
  dealii::DynamicSparsityPattern local_dsp(total_number_of_dofs_on_level, total_number_of_dofs_on_level);
  std::vector<DofNumber> dof_indices(get_local_problem()->base_problem.fe.dofs_per_cell);
  for(auto it = get_local_problem()->base_problem.dof_handler.begin(); it != get_local_problem()->base_problem.dof_handler.end(); it++) {
    if(it->at_boundary(surface)) {
      it->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < dof_indices.size(); i++) {
        dof_indices[i] += first_own_index;
      }
      constraints.add_entries_local_to_global(dof_indices, local_dsp);
    }
  }

  for(unsigned int i = 0; i < 6; i++) {
    if((i != surface) && (!are_opposing_sites(surface, i))) {
      if(is_hsie_surface[i]) {
        get_local_problem()->surfaces[i]->fill_sparsity_pattern_for_boundary_id(i, surface_first_dofs[i], &constraints, &local_dsp);
      }
    }
  }

  // Extract all couplings into a vector
  const unsigned int n_entries = local_dsp.n_nonzero_elements();
  DofNumber * rows = new DofNumber[n_entries];
  DofNumber * cols = new DofNumber[n_entries];
  unsigned int counter = 0;
  for(auto it = local_dsp.begin(); it != local_dsp.end(); it++) {
    rows[counter] = it->row();
    cols[counter] = it->column();
    counter ++;
  }
  // Send the vector via MPI to the partner
  unsigned long send_temp = n_entries;
  std::pair<bool, unsigned int> partner_data = GlobalMPI.get_neighbor_for_interface(get_direction_for_boundary_id(surface));
  unsigned int partner_index = partner_data.second;
  MPI_Sendrecv_replace(&send_temp, 1, MPI::UNSIGNED_LONG, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  if(send_temp != n_entries) {
    std::cout << "Size missmatch in fill_dsp_over_mpi" << std::endl;
  }
  MPI_Sendrecv_replace(rows, n_entries, MPI::UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  MPI_Sendrecv_replace(cols, n_entries, MPI::UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  // In the received vectors, replace the boundary dof ids of the other process with the own.
  for(unsigned int i = 0; i < coupling_dofs[surface].size(); i++) {
    for(unsigned int j = 0; j < n_entries; j++) {
      if(rows[j] == coupling_dofs[surface][i].second) {
        rows[j] = coupling_dofs[surface][i].first;
      }
      if(cols[j] == coupling_dofs[surface][i].second) {
        cols[j] = coupling_dofs[surface][i].first;
      }
    }
  }
  // Fill the argument dsp with the values received over MPI
  for(unsigned int i = 0; i < n_entries; i++) {
    in_dsp->add(rows[i], cols[i]);
  }
}

std::vector<ComplexNumber> NonLocalProblem::UpperBlockProductAfterH() {
  IndexSet is = lower_interface_dofs;
  setChildRhsComponentsFromU();
  child->solve();
  child->update_mismatch_vector(compute_upper_interface_id());
  
  std::vector<ComplexNumber> ret(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[is.nth_index_in_set(i) - first_own_index];
  }
  std::cout << "In rank " << rank << " Upper Block Product Norm is " << l2_norm_of_vector(ret) << std::endl;
  return ret;
}

std::vector<ComplexNumber> NonLocalProblem::LowerBlockProduct() {
  setChildSolutionComponentsFromU();
  child->update_mismatch_vector(compute_upper_interface_id());
  std::vector<ComplexNumber> ret(upper_interface_dofs.n_elements());
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[upper_interface_dofs.nth_index_in_set(i) - first_own_index + child->first_own_index];
  }
  std::cout << "In rank " << rank << " Lower Block Product Norm is " << l2_norm_of_vector(ret) << " and rhs_mismatch norm " << child->rhs_mismatch.l2_norm() << std::endl;
  return ret;
}

void NonLocalProblem::setSolutionFromVector(Vec x_in) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  VecGetValues(x_in, own_dofs.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < n_own_dofs; i++) {
    u[i] = values[i]; 
  }
  delete[] values;
}

void NonLocalProblem::setChildSolutionComponentsFromU() {
  for(unsigned int i = 0; i < get_local_problem()->base_problem.n_dofs; i++) {
    child->solution[child->first_own_index + i] = u[i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface] && child->is_hsie_surface[surface]) {
      for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
        child->solution[child->surface_first_dofs[surface] + i] = u[surface_first_dofs[surface] - first_own_index + i];
      }
    } else {
      if(child->is_hsie_surface[surface]) {
        std::vector<InterfaceDofData> vec = get_local_problem()->surfaces[surface]->get_dof_association();
        std::sort(vec.begin(), vec.end(), compareDofDataByGlobalIndex);
        unsigned int index = 0;
        for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
          while(vec[index].index < i){
            index++;
          }
          if(i != vec[index].index){
            child->solution[child->surface_first_dofs[surface] + i] = 0;
          } 
        }
      }
    }
  }
  child->solution.compress(VectorOperation::insert);
}

void NonLocalProblem::setChildRhsComponentsFromU() {
  for(unsigned int i = 0; i < get_local_problem()->base_problem.n_dofs; i++) {
    child->rhs[child->first_own_index + i] = u[i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface] && child->is_hsie_surface[surface]) {
      for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
        child->rhs[child->surface_first_dofs[surface] + i] = u[surface_first_dofs[surface] - first_own_index + i];
      }
    } else {
      if(child->is_hsie_surface[surface]) {
        for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
          // child->rhs[child->surface_first_dofs[surface] + i] = 0;
        }
      }
    }
  }
  child->rhs.compress(VectorOperation::insert);
}

DofOwner NonLocalProblem::get_dof_owner(unsigned int dof) {
  DofOwner ret;
  if(dof < first_own_index || dof > first_own_index + n_own_dofs) {
    std::cout << "get_dof_data was called for a dof that is not locally owned" << std::endl;
    return ret;
  }
  ret.owner = rank;
  ret.is_boundary_dof = dof > surface_first_dofs[0];
  if(ret.is_boundary_dof) {
    for(unsigned int i = 0; i < 5; i++) {
      if(dof < surface_first_dofs[i+1]) {
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
  return (dof >= first_own_index && dof < first_own_index + n_own_dofs);
}

bool NonLocalProblem::validate_constraints_range() {
  bool all_valid = true;
  dealii::IndexSet allowed_dofs = own_dofs;
  if(rank != 0) {
    for(auto it = index_sets_per_process[rank -1].begin(); it != index_sets_per_process[rank -1].end(); it++) allowed_dofs.add_index(*it);
  }
  if(rank != GlobalParams.Blocks_in_z_direction-1) {
    for(auto it = index_sets_per_process[rank +1].begin(); it != index_sets_per_process[rank +1].end(); it++) allowed_dofs.add_index(*it);
  }
  
  for(auto it = constraints.get_lines().begin(); it != constraints.get_lines().end(); it++) {
    if(! allowed_dofs.is_element(it->index)) {
      all_valid = false;
      break;
    }
    for(auto entry  = it->entries.begin(); entry != it->entries.end(); entry++) {
      if(! allowed_dofs.is_element(entry->first)) all_valid = false;
      break;
    }

  }
  return all_valid;
}
