#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
#include "../Helpers/staticfunctions.h"
#include "HierarchicalProblem.h"
#include "LocalProblem.h"
#include "../Core/NumericProblem.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
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
    is_hsie_surface[4] = true;
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

  std::vector<DofIndexAndOrientationAndPosition> from_surface =
      get_local_problem()->surfaces[surface]->get_dof_association();
  std::vector<DofIndexAndOrientationAndPosition> from_inner_problem =
      get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
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
    constraints.add_line(from_inner_problem[line].index);
    ComplexNumber value = { 0, 0 };
    if (from_inner_problem[line].orientation
        == from_surface[line].orientation) {
      value.real(1.0);
    } else {
      value.real(-1.0);
    }
    constraints.add_entry(from_inner_problem[line].index,
        from_surface[line].index + surface_first_dofs[surface], value);
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
  std::vector<DofIndexAndOrientationAndPosition> from_inner_problem =
      get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(surface);
  unsigned int n_dofs_on_surface = from_inner_problem.size();

  unsigned int lowest = coupling_dofs.size();
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
    indices[i] = from_inner_problem[i].index + local_indices.nth_index_in_set(0);
    coupling_dofs.emplace_back(indices[i], 0);
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
    coupling_dofs[lowest + i].second = indices[i];
  }
  if(!equals) {
    std::cout << "There was a data comparison error in NonlocalProblem::make_constraints_for_non_hsie_surface" << std::endl;
  }
  for(unsigned int i = 0; i < n_dofs_on_surface; i++) {
    constraints.add_line(from_inner_problem[i].index);
    double val = 1;
    if(orientations[i] == from_inner_problem[i].orientation) {
      val = -1;
    }
    constraints.add_entry(from_inner_problem[i].index, indices[i], val);
  }
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
  dealii::IndexSet is;
  is.set_size(total_number_of_dofs_on_level);
  is.add_range(0, total_number_of_dofs_on_level);
  constraints.reinit(is);
  coupling_dofs.clear();
  // couple surface dofs with inner ones.
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]){
      make_constraints_for_hsie_surface(surface);
    } else {
      make_constraints_for_non_hsie_surface(surface);
    }
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 1: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      if ( is_hsie_surface[i] && is_hsie_surface[j]) {
      surface_to_surface_constraints.reinit(is);
      bool opposing = ((i % 2) == 0) && (i + 1 == j);
      if (!opposing) {
        std::vector<DofIndexAndOrientationAndPosition> lower_face_dofs =
            get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<DofIndexAndOrientationAndPosition> upper_face_dofs =
            get_local_problem()->surfaces[j]->get_dof_association_by_boundary_id(i);
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
          unsigned int dof_a = lower_face_dofs[dof].index
              + surface_first_dofs[i];
          unsigned int dof_b = upper_face_dofs[dof].index
              + surface_first_dofs[j];
          ComplexNumber value = { 0, 0 };
          if (lower_face_dofs[dof].orientation
              == upper_face_dofs[dof].orientation) {
            value.real(1.0);
          } else {
            value.real(-1.0);
          }
          surface_to_surface_constraints.add_line(dof_a);
          surface_to_surface_constraints.add_entry(dof_a, dof_b, value);
        }
      }
      constraints.merge(surface_to_surface_constraints,
        dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins);
      }
    }
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 2: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  get_local_problem()->base_problem.make_constraints(&constraints, local_indices.nth_index_in_set(0), local_indices);
  print_info("LocalProblem::make_constraints", "Constraints after phase 3: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  print_info("NonLocalProblem::make_constraints", "End");
}

void NonLocalProblem::assemble() {
  print_info("NonLocalProblem::assemble", "Start");

  get_local_problem()->base_problem.assemble_system(local_indices.nth_index_in_set(0), &constraints, matrix, system_rhs);
  for(unsigned int i = 0; i< 6; i++) {
    print_info("NonLocalProblem::assemble", "assemble surface " + std::to_string(i));
    if(is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_matrix(matrix, system_rhs, surface_first_dofs[i], is_hsie_surface, &constraints);
    }
  }
  MPI_Barrier(GlobalMPI.communicators_by_level[local_level]);
  matrix->compress(dealii::VectorOperation::add);
  system_rhs->compress(dealii::VectorOperation::add);
  solution.compress(dealii::VectorOperation::add);
  child->assemble();
  dof_orientations_identical = get_incoming_dof_orientations();
  print_info("NonLocalProblem::assemble", "End");
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
    solution(first_own_index + i) = child->solution(child->first_own_index + i);
  }
}

void NonLocalProblem::apply_sweep(Vec x_in, Vec x_out) {
  print_info("NonLocalProblem::apply_sweep", "Start");
  MPI_Barrier(MPI_COMM_WORLD);
  ComplexNumber * values = new ComplexNumber[local_indices.n_elements()];
  VecGetValues(x_in, local_indices.n_elements(), locally_owned_dofs_index_array, values);
  for(unsigned int i = 0; i < local_indices.n_elements(); i++) {
    solution(local_indices.nth_index_in_set(i)) = values[i];
  }
  MPI_Barrier(MPI_COMM_WORLD);
  receive_local_upper_dofs();
  H_inverse();
  propagate_up();
  send_local_lower_dofs();
  
  MPI_Barrier(MPI_COMM_WORLD);
  H_inverse();
  propagate_up();
  
  MPI_Barrier(MPI_COMM_WORLD);
  receive_local_lower_dofs();
  H_inverse();
  propagate_up();
  send_local_upper_dofs();
  MPI_Barrier(MPI_COMM_WORLD);
  for(unsigned int i = 0; i < local_indices.n_elements(); i++) {
     values[i] = solution[local_indices.nth_index_in_set(i)];
  }
  VecSetValues(x_out, local_indices.n_elements(), locally_owned_dofs_index_array, values, INSERT_VALUES);
  VecAssemblyBegin(x_out);
  VecAssemblyEnd(x_out);
  MPI_Barrier(MPI_COMM_WORLD);
  delete[] values;
  print_info("NonLocalProblem::apply_sweep", "End");
}

void NonLocalProblem::reinit() {
  print_info("Nonlocal reinit", "Reinit starting");
  make_constraints();
  generate_sparsity_pattern();
  std::vector<DofCount> rows_per_process;
  std::vector<DofCount> columns_per_process;
  const DofCount max_couplings = get_local_problem()->base_problem.dof_handler.max_couplings_between_dofs();
  for(unsigned int i = 0; i < index_sets_per_process.size(); i++) {
    rows_per_process.push_back(index_sets_per_process[i].n_elements());
    columns_per_process.push_back(max_couplings);
  }
  matrix->reinit(GlobalMPI.communicators_by_level[local_level], total_number_of_dofs_on_level, total_number_of_dofs_on_level, local_indices.n_elements(), local_indices.n_elements(), 150);
  system_rhs = new dealii::PETScWrappers::MPI::Vector(local_indices, GlobalMPI.communicators_by_level[local_level]);
  solution.reinit(local_indices, GlobalMPI.communicators_by_level[local_level]);
  constraints.close();
  child->reinit();
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  child->initialize();
  for(unsigned int i = 0; i < 6; i++) {
    surface_dof_associations[i] = get_local_problem()->surfaces[i]->get_dof_association();
    for(unsigned int j = 0; j < surface_dof_associations[i].size(); j++) {
      surface_dof_index_vectors[i].push_back(first_own_index + surface_dof_associations[i][j].index);
    }
  }
  initialize_own_dofs();
  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();
  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
}

void NonLocalProblem::generate_sparsity_pattern() {
  dsp.reinit( total_number_of_dofs_on_level, total_number_of_dofs_on_level, local_indices);
  DofNumber first_index = local_indices.nth_index_in_set(0);
  get_local_problem()->base_problem.make_sparsity_pattern(&dsp, first_index , &constraints);
  first_index += get_local_problem()->base_problem.n_dofs;
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_sparsity_pattern(&dsp,
          first_index, &constraints);
      first_index += get_local_problem()->surfaces[i]->dof_counter;
    } 
  }
  for(unsigned int i=0;i<coupling_dofs.size(); i++) {
    dsp.add(coupling_dofs[i].first, coupling_dofs[i].second);
  }
  dsp.compress();
}

void NonLocalProblem::initialize_index_sets() {
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(
      GlobalMPI.communicators_by_level[local_level]);
  rank = dealii::Utilities::MPI::this_mpi_process(
      GlobalMPI.communicators_by_level[local_level]);
  index_sets_per_process =
      dealii::Utilities::MPI::create_ascending_partitioning(
      GlobalMPI.communicators_by_level[local_level], n_own_dofs);
  local_indices = index_sets_per_process[rank];
  total_number_of_dofs_on_level = 0;
  for (unsigned int i = 0; i < n_procs_in_sweep; i++) {
    total_number_of_dofs_on_level += index_sets_per_process[i].n_elements();
  }
  DofCount n_inner_dofs =
      this->get_local_problem()->base_problem.dof_handler.n_dofs()
          + local_indices.nth_index_in_set(0);
  surface_first_dofs.push_back(n_inner_dofs);
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      n_inner_dofs += this->get_local_problem()->surfaces[i]->dof_counter;
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
  first_own_index = local_indices.nth_index_in_set(0);
  lower_interface_dofs = compute_interface_dof_set(lower_sweeping_interface_id);
  upper_interface_dofs = compute_interface_dof_set(upper_sweeping_interface_id);
  cached_lower_values.resize(lower_interface_dofs.n_elements());
  cached_upper_values.resize(upper_interface_dofs.n_elements());
  locally_owned_dofs_index_array = new PetscInt[local_indices.n_elements()];
  get_petsc_index_array_from_index_set(locally_owned_dofs_index_array, local_indices);
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
        if(is_hsie_surface[i]) {
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
        ret.add_index(current[j].index + this->first_own_index);
      }      
    } else {
      if(i != opposing_interface_id) {
        if(is_hsie_surface[i]) {
          std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(i);
          for(unsigned int j = 0; j < current.size(); j++) {
            ret.add_index(current[j].index + this->first_own_index);
          }
        }
      }
    }
  }
  return ret;
}

unsigned int NonLocalProblem::compute_own_dofs() {
  surface_first_dofs.clear();
  DofCount ret = get_local_problem()->base_problem.dof_handler.n_dofs();
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    if (is_hsie_surface[i]) {
      ret += this->get_local_problem()->surfaces[i]->dof_counter;
    }
    if (i != 5) {
      surface_first_dofs.push_back(ret);
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

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {
  NumericVectorLocal ret(upper_interface_dofs.n_elements());
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    ret[i] = solution[upper_interface_dofs.nth_index_in_set(i)];
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::extract_local_lower_dofs() {
  NumericVectorLocal ret(lower_interface_dofs.n_elements());
  
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    ret[i] = solution[lower_interface_dofs.nth_index_in_set(i)];
  }
  return ret;
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

void NonLocalProblem::send_local_lower_dofs() {
  if(is_lowest_in_sweeping_direction()) {
    return;
  }
  reinit_mpi_cache();
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  NumericVectorLocal data_temp = extract_local_lower_dofs();
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    mpi_cache[i] = data_temp[i];
  }
  MPI_Send(&mpi_cache[0], lower_interface_dofs.n_elements(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::reinit_mpi_cache() {
  const unsigned int mpi_element_count = std::max(upper_interface_dofs.n_elements(), lower_interface_dofs.n_elements());
  mpi_cache = new ComplexNumber[mpi_element_count];
  for(unsigned int i = 0; i < mpi_element_count; i++) {
    mpi_cache[i] = 0;
  }
}

void NonLocalProblem::receive_local_lower_dofs() {
  if(is_lowest_in_sweeping_direction()) {
    return;
  }
  reinit_mpi_cache();
  const unsigned int n_elements = lower_interface_dofs.n_elements();
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], n_elements, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < n_elements; i++) {
    cached_lower_values[i] = mpi_cache[i] * (dof_orientations_identical[i] ? 1.0 : -1.0);
  }
  child->set_boundary_values(get_lower_boundary_id_for_sweeping_direction(sweeping_direction), cached_lower_values);
}

void NonLocalProblem::receive_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  reinit_mpi_cache();
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  MPI_Recv(&mpi_cache[0], upper_interface_dofs.n_elements(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    cached_upper_values[i] = mpi_cache[i] * (dof_orientations_identical[i] ? 1.0 : -1.0);
  }
  child->set_boundary_values(get_upper_boundary_id_for_sweeping_direction(sweeping_direction), cached_upper_values);
}

void NonLocalProblem::send_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  reinit_mpi_cache();
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  NumericVectorLocal data_temp = extract_local_upper_dofs();
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    mpi_cache[i] = data_temp[i];
  }
  MPI_Send(&mpi_cache[0], upper_interface_dofs.n_elements(), MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

auto NonLocalProblem::set_boundary_values(BoundaryId b_id, std::vector<ComplexNumber> dof_values) -> void {
  std::vector<DofNumber> indices;
  for(unsigned int i = 0; i < surface_index_sets[b_id].n_elements(); i++) {
    indices.push_back(surface_index_sets[b_id].nth_index_in_set(i));
    (*system_rhs)[surface_index_sets[b_id].nth_index_in_set(i)] = dof_values[i];
  }
  system_rhs->compress(dealii::VectorOperation::insert);
  child->set_boundary_values(b_id, dof_values);
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