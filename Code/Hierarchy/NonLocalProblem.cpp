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

#include <iterator>
#include <vector>


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
  HierarchicalProblem(local_level),
  sc(GlobalParams.GMRES_max_steps, GlobalParams.Solver_Precision, true, true),
  solver(sc, GlobalMPI.communicators_by_level[local_level], dealii::PETScWrappers::SolverGMRES::AdditionalData(GlobalParams.GMRES_Steps_before_restart))
{
    if(local_level > 1) {
    child = new NonLocalProblem(local_level - 1);
    } else {
    child = new LocalProblem();
    }
  switch (GlobalParams.HSIE_SWEEPING_LEVEL) {
    case 1:
      this->sweeping_direction = SweepingDirection::Z;
      break;
    case 2:
      if (local_level == 1) {
        this->sweeping_direction = SweepingDirection::Y;
      } else {
        this->sweeping_direction = SweepingDirection::Z;
      }
      break;
    case 3:
      if (local_level == 1) {
        this->sweeping_direction = SweepingDirection::X;
      } else {
        if (local_level == 2) {
          this->sweeping_direction = SweepingDirection::Y;
        } else {
          this->sweeping_direction = SweepingDirection::Z;
        }
      }
      break;
    default:
      this->sweeping_direction = SweepingDirection::Z;
      break;
  }

  for (unsigned int i = 0; i < 6; i++) {
    is_hsie_surface[i] = false;
    is_sweeping_hsie_surface[i] = false;
  }
  if (GlobalParams.Index_in_x_direction == 0) {
    is_hsie_surface[0] = true;
  }
  if (GlobalParams.Index_in_y_direction == 0) {
    is_hsie_surface[2] = true;
  }
  if (GlobalParams.Index_in_z_direction == 0) {
    is_hsie_surface[4] = false;
  }
  if (GlobalParams.Index_in_x_direction
      == GlobalParams.Blocks_in_x_direction - 1) {
    is_hsie_surface[1] = true;
  }
  if (GlobalParams.Index_in_y_direction
      == GlobalParams.Blocks_in_y_direction - 1) {
    is_hsie_surface[3] = true;
  }
  if (GlobalParams.Index_in_z_direction
      == GlobalParams.Blocks_in_z_direction - 1) {
    is_hsie_surface[5] = true;
  }
  if (GlobalParams.HSIE_SWEEPING_LEVEL == local_level + 1) {
    is_hsie_surface[4] = true;
    is_hsie_surface[5] = true;
  }
  if (GlobalParams.HSIE_SWEEPING_LEVEL == local_level + 2) {
    is_hsie_surface[2] = true;
    is_hsie_surface[3] = true;
    is_hsie_surface[4] = true;
    is_hsie_surface[5] = true;
  }
  n_own_dofs = 0;
  matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
  communicate_sweeping_direction(sweeping_direction);
}

void NonLocalProblem::init_solver_and_preconditioner() {
  print_info("NonLocalProblem::init_solver_and_preconditioner", "Start");
  print_info("NonLocalProblem::init_solver_and_preconditioner", "Init solver and pc on level " + std::to_string(local_level), true, LoggingLevel::PRODUCTION_ONE);
  dealii::PETScWrappers::PreconditionNone pc_none;
  pc_none.initialize(*matrix);
  solver.initialize(pc_none);
  KSPGetPC(solver.solver_data->ksp, &pc);
  PCSetType(pc,PCSHELL);
  pc_create(&shell, this);
  PCShellSetApply(pc,pc_apply);
  PCShellSetContext(pc, (void*) &shell);
  KSPSetPC(solver.solver_data->ksp, pc);
  print_info("NonLocalProblem::init_solver_and_preconditioner", "End");
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
  delete system_rhs;
  delete[] mpi_cache;
}

void NonLocalProblem::make_constraints_for_hsie_surface(unsigned int surface) {

  std::vector<DofIndexAndOrientationAndPosition> from_surface = get_local_problem()->surfaces[surface]->get_dof_association();
  std::vector<DofIndexAndOrientationAndPosition> from_inner_problem = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
  if (from_surface.size() != from_inner_problem.size()) {
    std::cout << "Warning: Size mismatch in make_constraints for surface "
        << surface << ": Inner: " << from_inner_problem.size()
        << " != Surface:" << from_surface.size() << "." << std::endl;
  }
  for (unsigned int line = 0; line < from_inner_problem.size(); line++) {
    if (!areDofsClose(from_inner_problem[line], from_surface[line])) {
      std::cout << "Error in face to inner_coupling. Positions are inner: "
          << from_inner_problem[line].position << " and surface: "
          << from_surface[line].position << std::endl;
    }
    constraints.add_line(from_inner_problem[line].index + first_own_index);
    ComplexNumber value = { 0, 0 };
    if (from_inner_problem[line].orientation == from_surface[line].orientation) {
      value.real(1.0);
    } else {
      value.real(-1.0);
    }
    constraints.add_entry(from_inner_problem[line].index + first_own_index, from_surface[line].index + surface_first_dofs[surface], value);
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

void NonLocalProblem::make_constraints_for_non_hsie_surface(unsigned int surface) {
  if(GlobalParams.Index_in_z_direction == 0 && surface == 4) {

  } else {
    std::vector<DofIndexAndOrientationAndPosition> from_inner_problem = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
    unsigned int n_dofs_on_surface = from_inner_problem.size();
    std::pair<bool, unsigned int> partner_data = GlobalMPI.get_neighbor_for_interface(get_direction_for_boundary_id(surface));
    if(!partner_data.first) {
      std::cout << "There was an error finding the partner process in NonlocalProblem::make_constraints_for_non_hsie_surface" << std::endl;
    }
    unsigned int partner_index = partner_data.second;
    bool * orientations = new bool[n_dofs_on_surface];
    double * x_values = new double[n_dofs_on_surface];
    double * y_values = new double[n_dofs_on_surface];
    double * z_values = new double[n_dofs_on_surface];
    DofNumber * indices = new unsigned int[n_dofs_on_surface];
    for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
      orientations[i] = from_inner_problem[i].orientation;
      x_values[i] = from_inner_problem[i].position[0];
      y_values[i] = from_inner_problem[i].position[1];
      z_values[i] = from_inner_problem[i].position[2];
      indices[i] = from_inner_problem[i].index + first_own_index;
      coupling_dofs[surface].emplace_back(indices[i], 0);
    }
    print_info("NonLocalProblem::make_constraints_for_non_hsie_surface", std::to_string(rank) + " connecting to " + std::to_string(partner_index), false, DEBUG_ALL);
    MPI_Sendrecv_replace(orientations, n_dofs_on_surface, MPI::BOOL, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    MPI_Sendrecv_replace(x_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    MPI_Sendrecv_replace(y_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    MPI_Sendrecv_replace(z_values, n_dofs_on_surface, MPI_DOUBLE, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    MPI_Sendrecv_replace(indices, n_dofs_on_surface, MPI_UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
    bool equals = true;
    for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
      // if(orientations[i] != from_inner_problem[i].orientation) equals = false;
      if(x_values[i] != from_inner_problem[i].position[0]) equals = false;
      if(y_values[i] != from_inner_problem[i].position[1]) equals = false;
      if(z_values[i] != from_inner_problem[i].position[2]) equals = false;
      coupling_dofs[surface][i].second = indices[i];
    }
    if(!equals) {
      std::cout << "There was a data comparison error in NonlocalProblem::make_constraints_for_non_hsie_surface" << std::endl;
    }
    for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
      double coupling_value = (orientations[i] != from_inner_problem[i].orientation) ? -1.0: 1.0;
      constraints.add_line(coupling_dofs[surface][i].first);
      constraints.add_entry(coupling_dofs[surface][i].first, coupling_dofs[surface][i].second, coupling_value);
    }
    
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

void NonLocalProblem::make_sparsity_pattern_for_surface(unsigned int surface, DynamicSparsityPattern * dsp) {
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
              couplings[index_in_coupling].push_back( local_dofs[j]);
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
    for(unsigned int j = 1; j < max_n_couplings+1; j++) {
      if(cache[j] != -1) {
        for(unsigned int local_dof_index = 0; local_dof_index < couplings[i].size(); local_dof_index++) {
          dsp->add(couplings[i][local_dof_index], cache[j]);
        }
      }
    }
  }
  fill_dsp_over_mpi(surface, dsp);
}

std::vector<bool> NonLocalProblem::get_incoming_dof_orientations() {
  std::vector<bool> ret;
  std::vector<DofIndexAndOrientationAndPosition> from_lower_surface =
    get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(compute_lower_interface_id());
    std::vector<DofIndexAndOrientationAndPosition> from_upper_surface =
    get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(compute_upper_interface_id());
  for(unsigned int i = 0; i < from_lower_surface.size(); i++) {
    if(from_lower_surface[i].orientation == from_upper_surface[i].orientation) {
      ret.push_back(true);
    } else {
      ret.push_back(false);
    }
  }
  return ret;
}

void NonLocalProblem::make_constraints() {
  print_info("NonLocalProblem::make_constraints", "Start");
  IndexSet total_dofs_global(total_number_of_dofs_on_level);
  total_dofs_global.add_range(0,total_number_of_dofs_on_level);
  constraints.reinit(total_dofs_global);
  for(unsigned int i = 0; i < 6; i++) { 
    coupling_dofs[i].clear();
  };
  // couple surface dofs with inner ones.
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]){
      make_constraints_for_hsie_surface(surface);
    } else {
      make_constraints_for_non_hsie_surface(surface);
    }
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 1: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  
  // Edge2Edge Dofs.
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      if ( is_hsie_surface[i] && is_hsie_surface[j]) {
      surface_to_surface_constraints.reinit(own_dofs);
      bool opposing = ((i % 2) == 0) && (i + 1 == j);
      if (!opposing) {
        std::vector<DofIndexAndOrientationAndPosition> lower_face_dofs = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<DofIndexAndOrientationAndPosition> upper_face_dofs = get_local_problem()->surfaces[j]->get_dof_association_by_boundary_id(i);
        if (lower_face_dofs.size() != upper_face_dofs.size()) {
          std::cout << "ERROR: There was a edge dof count error!" << std::endl
              << "Surface " << i << " offers " << lower_face_dofs.size()
              << " dofs, " << j << " offers " << upper_face_dofs.size() << "."
              << std::endl;
        }
        for (unsigned int dof = 0; dof < lower_face_dofs.size(); dof++) {
          if (!areDofsClose(lower_face_dofs[dof], upper_face_dofs[dof])) {
            std::cout << "Error in face to face_coupling. Positions are lower: "
                << lower_face_dofs[dof].position << " and upper: "
                << upper_face_dofs[dof].position << std::endl;
          }
          unsigned int dof_a = lower_face_dofs[dof].index + surface_first_dofs[i];
          unsigned int dof_b = upper_face_dofs[dof].index + surface_first_dofs[j];
          ComplexNumber value = { 0, 0 };
          if (lower_face_dofs[dof].orientation == upper_face_dofs[dof].orientation) {
            value.real(1.0);
          } else {
            value.real(-1.0);
          }
          surface_to_surface_constraints.add_line(dof_a);
          surface_to_surface_constraints.add_entry(dof_a, dof_b, value);
        }
      }
      constraints.merge(surface_to_surface_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins, true);
      }
    }
  }

  print_info("LocalProblem::make_constraints", "Constraints after phase 2: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  // From inner problem
  get_local_problem()->base_problem.make_constraints(&constraints, own_dofs.nth_index_in_set(0), own_dofs);
  print_info("LocalProblem::make_constraints", "Constraints after phase 3: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  print_info("NonLocalProblem::make_constraints", "End");
}

void NonLocalProblem::assemble() {
  get_local_problem()->base_problem.assemble_system(first_own_index, &constraints, matrix, system_rhs);
  for(unsigned int i = 0; i< 6; i++) {
    if(is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_matrix(matrix, system_rhs, surface_first_dofs[i], is_hsie_surface, &constraints);
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  system_rhs->compress(dealii::VectorOperation::add);
  solution.compress(dealii::VectorOperation::add);
  child->assemble();
  dof_orientations_identical = get_incoming_dof_orientations();
}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(n_own_dofs);
  return ret;
}

void NonLocalProblem::solve() {
  dealii::PETScWrappers::MPI::Vector rhs = *system_rhs;
  KSPSetConvergenceTest(solver.solver_data->ksp, &convergence_test, reinterpret_cast<void *>(&sc),nullptr);
  KSPSetTolerances(solver.solver_data->ksp, 0.000001, 1.0, 1000, 30);
  KSPSetUp(solver.solver_data->ksp);
  PetscErrorCode ierr = KSPSolve(solver.solver_data->ksp, rhs, solution);
  if(ierr != 0) {
    std::cout << "Error code from Petsc: " << std::to_string(ierr) << std::endl;
    throw new ExcPETScError(ierr);
  }
}

void NonLocalProblem::propagate_up(){
  const unsigned int count = get_local_problem()->base_problem.n_dofs;
  // Propagate the inner values. HSIE-dofs can be added later.
  for(unsigned int i = 0; i < count; i++) {
    u[first_own_index + i] = child->solution(child->first_own_index + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(child->is_hsie_surface[i] && is_hsie_surface[i]) {
      for(unsigned int j = 0; j < get_local_problem()->surfaces[i]->dof_counter; j++) {
        u[surface_first_dofs[i] + j] = child->solution(child->surface_first_dofs[i] + j);
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
  
  setChildSolutionComponentsFromU(); // sets rhs in child.
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
     values[i] = solution[own_dofs.nth_index_in_set(i)];
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
  dsp.reinit(total_number_of_dofs_on_level, total_number_of_dofs_on_level);
  make_constraints();
  constraints.close();
  generate_sparsity_pattern();
  
  system_rhs = new dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  u.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  solution.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  rhs_mismatch.reinit(own_dofs, GlobalMPI.communicators_by_level[local_level]);
  std::vector<DofCount> n_dofs_by_proc; 
  for(unsigned int i = 0; i < n_procs_in_sweep; i++) {
    n_dofs_by_proc.push_back(index_sets_per_process[i].n_elements());
  }
  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < coupling_dofs[i].size(); j++){
      if(coupling_dofs[i][j].first == 25620 ||coupling_dofs[i][j].second == 25620 ) {
        std::cout << " The dof you mentioned is a coupling dof: " << coupling_dofs[i][j].first << " to " << coupling_dofs[i][j].second << std::endl;
      }
    }
  }
  matrix->reinit(GlobalMPI.communicators_by_level[local_level], sp, n_dofs_by_proc, n_dofs_by_proc, rank, true);
  // matrix->reinit(GlobalMPI.communicators_by_level[local_level], total_number_of_dofs_on_level, total_number_of_dofs_on_level, n_own_dofs, n_own_dofs, 2000, false, 1000 );
  
  child->reinit();
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  child->initialize();
  for(unsigned int i = 0; i < 6; i++) {
    if(get_local_problem()->is_hsie_surface[i]) {
      surface_dof_associations[i] = get_local_problem()->surfaces[i]->get_dof_association();
      for(unsigned int j = 0; j < surface_dof_associations[i].size(); j++) {
        surface_dof_index_vectors[i].push_back(first_own_index + surface_dof_associations[i][j].index);
      }
    }
  }
  initialize_own_dofs();
  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();
  initialize_index_sets();
  surface_first_dofs.clear();
  unsigned int current = first_own_index + get_local_problem()->base_problem.n_dofs;
  for(unsigned int surface = 0; surface < 6; surface++) {
    surface_first_dofs.push_back(current);
    if(is_hsie_surface[surface] && get_local_problem()->is_hsie_surface[surface]) {
      current += get_local_problem()->surfaces[surface]->dof_counter;
    }
  }
  reinit();
  init_solver_and_preconditioner();
}

void NonLocalProblem::generate_sparsity_pattern() {
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
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(
      GlobalMPI.communicators_by_level[local_level]);
  rank = dealii::Utilities::MPI::this_mpi_process(
      GlobalMPI.communicators_by_level[local_level]);
  index_sets_per_process = dealii::Utilities::MPI::create_ascending_partitioning(GlobalMPI.communicators_by_level[local_level], n_own_dofs);
  own_dofs = index_sets_per_process[rank];
  total_number_of_dofs_on_level = 0;
  for (unsigned int i = 0; i < n_procs_in_sweep; i++) {
    total_number_of_dofs_on_level += index_sets_per_process[i].n_elements();
  }
  std::cout << "Rank " << rank << " has " << n_own_dofs << " dofs. The computed total is "<< total_number_of_dofs_on_level << std::endl;
  DofCount n_inner_dofs = get_local_problem()->base_problem.dof_handler.n_dofs() + own_dofs.nth_index_in_set(0);
  surface_first_dofs.push_back(n_inner_dofs);
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i] && get_local_problem()->is_hsie_surface[i]) {
      n_inner_dofs += get_local_problem()->surfaces[i]->dof_counter;
    }
    if (i != 5) {
      surface_first_dofs.push_back(n_inner_dofs);
    }
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
  lower_sweeping_interface_id = compute_lower_interface_id();
  upper_sweeping_interface_id = compute_upper_interface_id();
  first_own_index = own_dofs.nth_index_in_set(0);
  lower_interface_dofs = compute_interface_dof_set(lower_sweeping_interface_id);
  upper_interface_dofs = compute_interface_dof_set(upper_sweeping_interface_id);
  cached_lower_values.resize(lower_interface_dofs.n_elements());
  cached_upper_values.resize(upper_interface_dofs.n_elements());
  locally_owned_dofs_index_array = new PetscInt[own_dofs.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, own_dofs);
}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}

void NonLocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
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
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  dealii::IndexSet ret(total_number_of_dofs_on_level);
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(interface_id);
      for(unsigned int j = 0; j < current.size(); j++) {
        ret.add_index(current[j].index + first_own_index);
      }      
    } else {
      if(i != opposing_interface_id) {
        if(is_hsie_surface[i] && get_local_problem()->is_hsie_surface[i]) {
          std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(i);
          for(unsigned int j = 0; j < current.size(); j++) {
            ret.add_index(current[j].index + first_own_index);
          }
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

auto NonLocalProblem::communicate_sweeping_direction(SweepingDirection) -> void {
  child->communicate_sweeping_direction(sweeping_direction);
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
  mpi_cache = new ComplexNumber[n_elements];
  for(unsigned int i = 0; i < n_elements; i++) {
    mpi_cache[i] = 0;
  }
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
  child->rhs = 0;
  for(unsigned int i = 0; i < n_elements; i++) {
    child->rhs[lower_interface_dofs.nth_index_in_set(i) - first_own_index + child->first_own_index] = mpi_cache[i] * (dof_orientations_identical[i] ? 1.0 : -1.0);
  }
  child->solve();
  const unsigned int count = get_local_problem()->base_problem.n_dofs;
  for(unsigned int i = 0; i < count; i++) {
    u[first_own_index + i] -= child->solution(child->first_own_index + i);
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(child->is_hsie_surface[i] && is_hsie_surface[i]) {
      for(unsigned int j = 0; j < get_local_problem()->surfaces[i]->dof_counter; j++) {
        u[surface_first_dofs[i] + j] -= child->solution(child->surface_first_dofs[i] + j);
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
    u[upper_interface_dofs.nth_index_in_set(i)] -= mpi_cache[i] * (dof_orientations_identical[i] ? 1.0 : -1.0);
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

void NonLocalProblem::set_boundary_values(BoundaryId b_id, std::vector<ComplexNumber> dof_values) {
  for(unsigned int i = 0; i < surface_index_sets[b_id].n_elements(); i++) {
    (*system_rhs)[surface_index_sets[b_id].nth_index_in_set(i)] = dof_values[i];
  }
  system_rhs->compress(dealii::VectorOperation::insert);
  child->set_boundary_values(b_id, dof_values);
}

void NonLocalProblem::update_mismatch_vector() {
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
    ret[i] = rhs_mismatch[first_own_index +is.nth_index_in_set(i)];
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

auto NonLocalProblem::release_boundary_values(BoundaryId b_id) -> void {
  const unsigned int n_dofs = surface_dof_index_vectors[b_id].size();
  std::vector<ComplexNumber> values;
  for(unsigned int i = 0; i < n_dofs; i++) {
    values.emplace_back(0,0);
  }
  rhs.set(surface_dof_index_vectors[b_id], values);
  system_rhs->compress(dealii::VectorOperation::insert);
}

void NonLocalProblem::compute_solver_factorization() {
  this->child->compute_solver_factorization();
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
  BoundaryId bid = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = child->surface_index_sets[bid];
  setChildSolutionComponentsFromU();
  child->solve();
  child->update_mismatch_vector();
  
  std::vector<ComplexNumber> ret;
  ret.resize(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[is.nth_index_in_set(i)];
  }
  return ret;
}

std::vector<ComplexNumber> NonLocalProblem::LowerBlockProduct() {
  BoundaryId bid = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  IndexSet is = child->surface_index_sets[bid];
  setChildRhsComponentsFromU();
  child->update_mismatch_vector();
  std::vector<ComplexNumber> ret;
  ret.resize(is.n_elements());
  for(unsigned int i = 0; i < is.n_elements(); i++) {
    ret[i] = child->rhs_mismatch[is.nth_index_in_set(i)];
  }
  return ret;
}

void NonLocalProblem::setSolutionFromVector(Vec x_in) {
  ComplexNumber * values = new ComplexNumber[own_dofs.n_elements()];
  VecGetValues(x_in, own_dofs.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < own_dofs.n_elements(); i++) {
    u(own_dofs.nth_index_in_set(i)) = values[i]; 
  }
  delete[] values;
}

void NonLocalProblem::setChildSolutionComponentsFromU() {
  for(unsigned int i = 0; i < get_local_problem()->base_problem.n_dofs; i++) {
    child->solution[child->first_own_index + i] = u[first_own_index + i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface] && child->is_hsie_surface[surface]) {
      for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
        child->solution[child->surface_first_dofs[surface] + i] = u[surface_first_dofs[surface] + i];
      }
    } else {
      if(child->is_hsie_surface[surface]) {
        for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
          child->solution[child->surface_first_dofs[surface] + i] = 0;
        }
      }
    }
  }
}

void NonLocalProblem::setChildRhsComponentsFromU() {
  for(unsigned int i = 0; i < get_local_problem()->base_problem.n_dofs; i++) {
    child->rhs[child->first_own_index + i] = u[first_own_index + i];
  }
  for(unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface] && child->is_hsie_surface[surface]) {
      for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
        child->rhs[child->surface_first_dofs[surface] + i] = u[surface_first_dofs[surface] + i];
      }
    } else {
      if(child->is_hsie_surface[surface]) {
        for(unsigned int i = 0; i < get_local_problem()->surfaces[surface]->dof_counter; i++) {
          child->rhs[child->surface_first_dofs[surface] + i] = 0;
        }
      }
    }
  }
}