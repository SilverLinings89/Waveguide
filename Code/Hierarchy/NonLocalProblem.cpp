#include "NonLocalProblem.h"
#include "../Helpers/GeometryManager.h"
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

void get_petsc_index_array_from_index_set(PetscInt* in_array, dealii::IndexSet in_set) {
  for(unsigned int i = 0; i < in_set.n_elements(); i++) {
    in_array[i] = in_set.nth_index_in_set(i);
  }
}

PetscErrorCode pc_apply(PC in_pc, Vec x_in, Vec x_out) {
  VecCopy(x_in, x_out);
  SampleShellPC  *shell;
  PCShellGetContext(in_pc,(void**)&shell);

  // Extract the components from the input vector.
  std::vector<ComplexNumber> vals;
  vals.resize(shell->n_local_problem_indices);
   // ComplexNumber * vals = new ComplexNumber[shell->n_local_problem_indices];
  VecGetValues(x_in,shell->n_local_problem_indices, shell->parent_elements_vec_petsc, &vals[0]);
  
  // Set them in the child vector.
  shell->child->solution.set(shell->child_elements_vec, vals);

  // Solve one level below.
  // shell->child->solve();
  
  // Extract the components from the solution.
  for(unsigned int i = 0; i < shell->n_local_problem_indices; i++) {
    vals[i] = shell->child->solution[shell->child_elements_vec[i]];
  }

  // Set them in the parent vector.
  shell->parent->solution.set(shell->parent_elements_vec, vals);
  
  return 0;
}

PetscErrorCode pc_create(SampleShellPC *shell, HierarchicalProblem * parent, HierarchicalProblem * child)
{
  SampleShellPC  *newctx;
  PetscNew(&newctx);
  newctx->child = child;
  newctx->parent = parent;
  newctx->n_local_problem_indices = parent->get_local_problem()->base_problem.n_dofs;
  newctx->parent_elements_vec_petsc = new PetscInt[parent->get_local_problem()->base_problem.n_dofs];
  newctx->child_elements_vec_petsc = new PetscInt[child->get_local_problem()->base_problem.n_dofs];
  for(unsigned int i = 0; i < newctx->n_local_problem_indices; i++) {
    newctx->parent_elements_vec.push_back(i + parent->first_own_index);
    newctx->child_elements_vec.push_back(i + child->first_own_index);
    newctx->parent_elements_vec_petsc[i] = i + parent->first_own_index;
    newctx->child_elements_vec_petsc[i] = i + child->first_own_index;
  }
  shell = newctx;
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
  print_info("NonLocalProblem init solver and preconditioner", "start");
  std::cout << "Init solver and pc on level " << local_level << std::endl;
  dealii::PETScWrappers::PreconditionNone pc_none;
  pc_none.initialize(*matrix);
  solver.initialize(pc_none);
  KSPGetPC(solver.solver_data->ksp, &pc);
  PCSetType(pc,PCSHELL);
  pc_create(&shell, this, child);
  PCShellSetApply(pc,pc_apply);
  PCShellSetContext(pc, (void*) &shell);
  KSPSetPC(solver.solver_data->ksp, pc);
  print_info("NonLocalProblem init solver and preconditioner", "end");
}

NonLocalProblem::~NonLocalProblem() {
  delete matrix;
  delete system_rhs;
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
  }
  MPI_Sendrecv_replace(orientations, n_dofs_on_surface, MPI_C_BOOL, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
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
    constraints.add_entry(from_inner_problem[i].index, orientations[i], val);
  }
}

void NonLocalProblem::make_constraints() {
  std::cout << "Making constraints" << std::endl;
  dealii::IndexSet is;
  is.set_size(total_number_of_dofs_on_level);
  is.add_range(0, total_number_of_dofs_on_level);
  constraints.reinit(is);

  // couple surface dofs with inner ones.
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]){
      make_constraints_for_hsie_surface(surface);
    } else {
      make_constraints_for_non_hsie_surface(surface);
    }
  }
  std::cout << "Constraints after phase 1:" << constraints.n_constraints()
      << std::endl;
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
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
  std::cout << "Constraints after phase 2:" << constraints.n_constraints() << std::endl;
  get_local_problem()->base_problem.make_constraints(&constraints, local_indices.nth_index_in_set(0), local_indices);
  std::cout << "Constraints after phase 3:" << constraints.n_constraints() << std::endl;
  std::cout << "End Make Constraints." << std::endl;
}

void NonLocalProblem::assemble() {
  Position center = get_center();
  get_local_problem()->base_problem.assemble_system(local_indices.nth_index_in_set(0), &constraints, matrix, system_rhs);
  for(unsigned int i = 0; i< 6; i++) {
    if(is_hsie_surface[i]) {
      get_local_problem()->surfaces[i]->fill_matrix(matrix, system_rhs, surface_first_dofs[i], center, &constraints);
    }
  }
  child->assemble();
}

dealii::Vector<ComplexNumber> NonLocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(n_own_dofs);
  return ret;
}

void NonLocalProblem::solve() {
  receive_local_upper_dofs();
  H_inverse(solution, temp_solution);
  send_local_lower_dofs();

  H_inverse(solution, temp_solution);

  receive_local_lower_dofs();
  H_inverse(solution, temp_solution);
  send_local_upper_dofs();
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
  matrix->reinit(GlobalMPI.communicators_by_level[local_level], sp, rows_per_process, rows_per_process, rank);
  system_rhs = new dealii::PETScWrappers::MPI::Vector(local_indices, GlobalMPI.communicators_by_level[local_level]);
  constraints.close();
  print_info("Nonlocal reinit", "Reinit done");
}

void NonLocalProblem::initialize() {
  child->initialize();
  initialize_own_dofs();
  dofs_process_above = compute_upper_interface_dof_count();
  dofs_process_below = compute_lower_interface_dof_count();
  initialize_index_sets();
  reinit();
  init_solver_and_preconditioner();
}

void NonLocalProblem::generate_sparsity_pattern() {
  DynamicSparsityPattern dsp = { total_number_of_dofs_on_level,
      total_number_of_dofs_on_level, local_indices };
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
  sp.copy_from(dsp);
  print_info("Nonlocal Problem generate dsp rows", dsp.n_rows());
  print_info("Nonlocal Problem generate dsp cols", dsp.n_cols());
  local_indices.print(std::cout);
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
  upper_interface_dofs = compute_interface_dof_set(compute_upper_interface_id(), compute_lower_interface_id());
  lower_interface_dofs = compute_interface_dof_set(compute_lower_interface_id(), compute_upper_interface_id());
}

LocalProblem* NonLocalProblem::get_local_problem() {
  return child->get_local_problem();
}

void NonLocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
}

DofCount NonLocalProblem::compute_interface_dofs(BoundaryId interface_id, BoundaryId opposing_interface_id) {
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

dealii::IndexSet NonLocalProblem::compute_interface_dof_set(BoundaryId interface_id, BoundaryId opposing_interface_id) {
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
  return compute_interface_dofs(compute_lower_interface_id(), compute_upper_interface_id());
}

auto NonLocalProblem::compute_upper_interface_dof_count() -> DofCount {
  return compute_interface_dofs(compute_upper_interface_id(), compute_lower_interface_id());
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

void NonLocalProblem::H_inverse(NumericVectorDistributed & src, NumericVectorDistributed & dst) {
  KSPSolve(solver.solver_data->ksp, src, dst);
  src = dst;
}

NumericVectorLocal NonLocalProblem::extract_local_upper_dofs() {
  NumericVectorLocal ret(upper_interface_dofs.n_elements());
  for(unsigned int i = 0; i < upper_interface_dofs.n_elements(); i++) {
    ret[i] = current_solution[upper_interface_dofs.nth_index_in_set(i)];
  }
  return ret;
}

NumericVectorLocal NonLocalProblem::extract_local_lower_dofs() {
  NumericVectorLocal ret(lower_interface_dofs.n_elements());
  for(unsigned int i = 0; i < lower_interface_dofs.n_elements(); i++) {
    ret[i] = current_solution[lower_interface_dofs.nth_index_in_set(i)];
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
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_below);
  NumericVectorLocal data_temp = extract_local_lower_dofs();
  for(unsigned int i = 0; i < dofs_process_above; i++) {
    data[i] = data_temp[i];
  }
  MPI_Send(&data[0], dofs_process_below, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::receive_local_lower_dofs() {
  if(is_lowest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_lower_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data =GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_below);
  MPI_Recv(&data[0], dofs_process_below, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < dofs_process_below; i++) {
    current_solution[lower_interface_dofs.nth_index_in_set(i)] = data[i];
  }
}

void NonLocalProblem::send_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_above);
  NumericVectorLocal data_temp = extract_local_upper_dofs();
  for(unsigned int i = 0; i < dofs_process_above; i++) {
    data[i] = data_temp[i];
  }
  MPI_Send(&data[0], dofs_process_above, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD);
}

void NonLocalProblem::receive_local_upper_dofs() {
  if(is_highest_in_sweeping_direction()) {
    return;
  }
  Direction communication_direction = get_upper_boundary_id_for_sweeping_direction(sweeping_direction);
  std::pair<bool, unsigned int> neighbour_data = GlobalMPI.get_neighbor_for_interface(communication_direction);
  ComplexNumber * data = new ComplexNumber(dofs_process_above);
  MPI_Recv(&data[0], dofs_process_above, MPI_C_DOUBLE_COMPLEX, neighbour_data.second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(unsigned int i = 0; i < dofs_process_above; i++) {
    current_solution[upper_interface_dofs.nth_index_in_set(i)] = data[i];
  }
}

auto NonLocalProblem::set_boundary_values(dealii::IndexSet local_indices, std::vector<ComplexNumber> dof_values) -> void {
  if(local_indices.n_elements() == dof_values.size()) {
    std::vector<unsigned int> indices;
    for(auto item: local_indices) {
      indices.push_back(item + first_own_index);
    }
    rhs.set(indices, dof_values);
    child->set_boundary_values(local_indices, dof_values);
  } else {
    std::cout << "Boundary values were passed incorrectly.";
  }
}

auto NonLocalProblem::release_boundary_values(dealii::IndexSet local_indices) -> void {
  std::vector<unsigned int> indices;
  std::vector<ComplexNumber> values;
  for(auto item: local_indices) {
    indices.push_back(item + first_own_index);
    values.push_back(0);
  }
  rhs.set(indices, values);
  child->release_boundary_values(local_indices);
}
