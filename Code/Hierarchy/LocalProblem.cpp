//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"

LocalProblem::LocalProblem() :
    HierarchicalProblem(0), base_problem() {
  base_problem.make_grid();

}

LocalProblem::~LocalProblem() {

}

void LocalProblem::solve(dealii::Vector<double> src,
    dealii::Vector<double> &dst) {
  dealii::Vector<double> inputb(n_own_dofs);
  for (unsigned int i = 0; i < n_own_dofs; i++) {
    inputb[i] = src(i);
  }

  solver.solve(inputb);

  for (unsigned int i = 0; i < n_own_dofs; i++) {
    dst[i] = inputb[i];
  }
}

void LocalProblem::initialize() {
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  std::complex<double> k0;
  std::map<dealii::Triangulation<2, 3>::cell_iterator,
      dealii::Triangulation<3, 3>::face_iterator> association;
  for (unsigned int side = 0; side < 6; side++) {
    std::cout << "Initializing surface " << side << " in local problem.";
    dealii::Triangulation<3> tria;
    tria.copy_triangulation(base_problem.triangulation);
    std::set<unsigned int> b_ids;
    b_ids.insert(side);
    switch (side) {
    case 0:
      dealii::GridTools::transform(Transform_0_to_5, tria);
      break;
    case 1:
      dealii::GridTools::transform(Transform_1_to_5, tria);
      break;
    case 2:
      dealii::GridTools::transform(Transform_2_to_5, tria);
      break;
    case 3:
      dealii::GridTools::transform(Transform_3_to_5, tria);
      break;
    case 4:
      dealii::GridTools::transform(Transform_4_to_5, tria);
      break;
    default:
      break;
    }
    association = dealii::GridGenerator::extract_boundary_mesh(tria,
        temp_triangulation, b_ids);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    switch (side) {
    case 0:
      surface_0 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 1:
      surface_1 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 2:
      surface_2 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 3:
      surface_3 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 4:
      surface_4 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 5:
      surface_5 = new HSIESurface(5, surf_tria, side, 0,
          GlobalParams.So_ElementOrder, k0, association);
      break;
    default:
      break;
    }
  }
  surfaces = new HSIESurface*[6];
  surfaces[0] = surface_0;
  surfaces[1] = surface_1;
  surfaces[2] = surface_2;
  surfaces[3] = surface_3;
  surfaces[4] = surface_4;
  surfaces[5] = surface_5;
}

void LocalProblem::generate_sparsity_pattern() {
}

unsigned int LocalProblem::compute_own_dofs() {
  unsigned int ret = 0;
  ret += base_problem.dof_handler.n_dofs();
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    ret += surfaces[i]->dof_counter;
    if (i != 5) {
      surface_first_dofs.push_back(ret);
    }
  }
  return ret;
}

void LocalProblem::assemble() {
  base_problem.assemble_system();
  matrix = new dealii::TrilinosWrappers::SparseMatrix(n_own_dofs, n_own_dofs,
      100);
  matrix->copy_from(base_problem.system_matrix);

  matrix = &base_problem.system_matrix;
}

void LocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
}

void LocalProblem::run() {
  assemble();
  // solve();
}

void LocalProblem::initialize_index_sets() {
  n_procs_in_sweep = dealii::Utilities::MPI::n_mpi_processes(
      GlobalMPI.communicators_by_level[1]);
  rank = dealii::Utilities::MPI::this_mpi_process(
      GlobalMPI.communicators_by_level[1]);
  unsigned int *all_dof_counts = new unsigned int[n_procs_in_sweep];
  MPI_Allgather(&this->n_own_dofs, 1, MPI_UINT16_T, all_dof_counts,
      n_procs_in_sweep, MPI_UINT16_T, GlobalMPI.communicators_by_level[1]);
  if (rank > 0) {
    dofs_process_below = all_dof_counts[rank - 1];
  }
  if (rank + 1 < n_procs_in_sweep) {
    dofs_process_above = all_dof_counts[rank + 1];
  }
}

void LocalProblem::apply_sweep(
    dealii::LinearAlgebra::distributed::Vector<double>) {

}

unsigned int LocalProblem::compute_lower_interface_dof_count() {
  // For local problems there are not interfaces.
  return 0;
}

unsigned int LocalProblem::compute_upper_interface_dof_count() {
  // For local problems there are not interfaces.
  return 0;
}

LocalProblem* LocalProblem::get_local_problem() {
  return this;
}

void LocalProblem::vmult(TrilinosWrappers::MPI::Vector &dst,
    const TrilinosWrappers::MPI::Vector &src) const {
  /**
   dealii::Vector<double> recv_buffer_above(dofs_process_above);
   dealii::Vector<double> recv_buffer_below(dofs_process_below);
   dealii::Vector<double> temp_own(n_own_dofs);
   dealii::Vector<double> temp_own_2(n_own_dofs);
   dealii::Vector<double> input(n_own_dofs);
   const MPI_Comm mpi_comm = GlobalMPI.communicators_by_level[1];
   for (unsigned int i = 0; i < n_own_dofs; i++) {
    input[i] = src[indices[i]];
  }
   if (rank + 1 == n_procs_in_sweep) {
   solver.solve(input);
   MPI_Send(&input[0], n_own_dofs, MPI_DOUBLE, rank - 1, 0, mpi_comm);
  } else {
   MPI_Recv(&recv_buffer_below[0], dofs_process_below, MPI_DOUBLE, rank + 1, 0, mpi_comm,
    MPI_STATUS_IGNORE);
    UpperProduct(recv_buffer_below, temp_own);
    input -= temp_own;
    if (rank != 0) {
      Hinv(input, temp_own);
   MPI_Send(&temp_own[0], n_own_dofs, MPI_DOUBLE, rank - 1, 0, mpi_comm);
    }
  }
  if (rank + 1 != GlobalParams.NumberProcesses) {
   for (unsigned int i = 0; i < n_own_dofs; i++) {
      temp_own[i] = input[i];
    }
    Hinv(temp_own, input);
  }
  if (rank == 0) {
   MPI_Send(&input[0], n_own_dofs, MPI_DOUBLE, rank + 1, 0, mpi_comm);
  } else {
   MPI_Recv(&recv_buffer_above[0], dofs_process_above, MPI_DOUBLE, rank - 1, 0, mpi_comm,
    MPI_STATUS_IGNORE);
    double recv_norm = 0.0;
   for (unsigned int i = 0; i < dofs_process_above; i++) {
      recv_norm += std::abs(recv_buffer_above[i]);
    }
    LowerProduct(recv_buffer_above, temp_own);
    Hinv(temp_own, temp_own_2);
    input -= temp_own_2;
    if (rank + 1 < GlobalParams.NumberProcesses) {
   MPI_Send(&input[0], n_own_dofs, MPI_DOUBLE, rank + 1, 0, mpi_comm);
    }
  }

   for (unsigned int i = 0; i < n_own_dofs; i++) {
    if (!fixed_dofs->is_element(indices[i])) {
      dst[indices[i]] = input[i];
    }
  }

  double delta = 0;
   for (unsigned int i = 0; i < n_own_dofs; i++) {
    if (!fixed_dofs->is_element(indices[i])) {
      delta += std::abs(dst[indices[i]] - src[indices[i]]);
    }
  }
   **/
}
