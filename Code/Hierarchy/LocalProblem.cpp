//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"
#include <iostream>
#include <fstream>
#include <deal.II/lac/solver_idr.h>

LocalProblem::LocalProblem() :
    HierarchicalProblem(0), base_problem() {
  base_problem.make_grid();

}

LocalProblem::~LocalProblem() {

}

void LocalProblem::solve(dealii::Vector<std::complex<double>> src,
    dealii::Vector<std::complex<double>> &dst) {
  dealii::Vector<std::complex<double>> inputb(n_own_dofs);
  for (unsigned int i = 0; i < n_own_dofs; i++) {
    inputb[i] = src(i);
  }

  // solver.solve(inputb);

  for (unsigned int i = 0; i < n_own_dofs; i++) {
    dst[i] = inputb[i];
  }

  // dealii::SolverID
}

void LocalProblem::initialize() {
  base_problem.setup_system();
  for (unsigned int side = 0; side < 6; side++) {

    dealii::Triangulation<2, 3> temp_triangulation;
    dealii::Triangulation<2> surf_tria;
    std::complex<double> k0(0, 1);
    std::cout << "Initializing surface " << side << " in local problem."
        << std::endl;
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
    dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation,
        b_ids);
    const unsigned int component = side / 2;
    auto temp_it = temp_triangulation.begin();
    double additional_coorindate = temp_it->center()[component];
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    switch (side) {
    case 0:
      surface_0 = new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
      break;
    case 1:
      surface_1 = new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
      break;
    case 2:
      surface_2 = new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
      break;
    case 3:
      surface_3 = new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
      break;
    case 4:
      surface_4 = new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
      break;
    case 5:
      surface_5 = new HSIESurface(5, surf_tria, side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate);
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
  for (unsigned int i = 0; i < 6; i++) {
    surfaces[i]->initialize();
  }

  std::cout << "Done" << std::endl;
  std::cout << "Initialize index sets" << std::endl;
  initialize_own_dofs();
  std::cout << "Number of local dofs: " << n_own_dofs << std::endl;
  std::cout << "End of Local Problem Initialize()." << std::endl;
}

void LocalProblem::generate_sparsity_pattern() {
}

void LocalProblem::validate() {
  std::cout << "Validate System: " << std::endl;
  std::cout << "N Rows: " << matrix->m() << std::endl;
  std::cout << "N Cols: " << matrix->n() << std::endl;
  std::cout << "Actually non-zero elements: "
      << matrix->n_actually_nonzero_elements(0.001) << std::endl;
  std::cout << "Base Problem non-zero elements: "
      << base_problem.system_matrix.n_nonzero_elements() << std::endl;
  std::cout << "Matrix l1 norm: " << matrix->l1_norm() << std::endl;
  for (unsigned int i = 0; i < matrix->n(); i++) {
    auto it = matrix->begin(i);
    auto end = matrix->end(i);
    unsigned int entries = 0;
    while (it != end) {
      if (it->value() != 0.0) {
        entries++;
      }
      it++;
    }
    if (entries == 0) {
      std::cout << "There were no entries in line " << i << std::endl;
    }

  }

}

unsigned int LocalProblem::compute_own_dofs() {
  std::cout << "Begin Compute own dofs: " << std::endl;
  surface_first_dofs.clear();
  unsigned int ret = base_problem.dof_handler.n_dofs();
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    ret += surfaces[i]->dof_counter;
    if (i != 5) {
      surface_first_dofs.push_back(ret);
    }
  }
  for (unsigned int i = 0; i < surface_first_dofs.size(); i++) {
    std::cout << surface_first_dofs[i] << " ";
  }
  std::cout << std::endl;
  return ret;
}

void LocalProblem::make_constraints() {
  std::cout << "Making constraints" << std::endl;
  dealii::IndexSet is;
  is.set_size(n_own_dofs);
  is.add_range(0, n_own_dofs);
  constraints.reinit(is);
  for (unsigned int surface = 0; surface < 6; surface++) {
    std::vector<unsigned int> from_surface =
        surfaces[surface]->get_dof_association();
    std::vector<unsigned int> from_inner_problem =
        base_problem.get_surface_dof_vector_for_boundary_id(surface);
    if (from_surface.size() != from_inner_problem.size()) {
      std::cout << "Warning: Size mismatch in make_constraints for surface "
          << surface << ": Inner: " << from_inner_problem.size()
          << " != Surface:" << from_surface.size() << "." << std::endl;
    }
    for (unsigned int line = 0; line < from_inner_problem.size(); line++) {
        constraints.add_line(from_inner_problem[line]);
        constraints.add_entry(from_inner_problem[line],
          from_surface[line] + surface_first_dofs[surface],
          std::complex<double>(-1, 0));
    }

  }

  // Only for point sorce example
  dealii::Point<3> center(0.0, 0.0, 0.0);
  std::vector<unsigned int> restrained_dofs =
      base_problem.dofs_for_cell_around_point(center);
  for (unsigned int i = 0; i < restrained_dofs.size(); i++) {
    constraints.add_line(restrained_dofs[i]);
    constraints.set_inhomogeneity(restrained_dofs[i],
        std::complex<double>(1, 0));
  }


  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      bool opposing = ((i % 2) == 0) && (i + 1 == j);
      if (!opposing) {
        std::vector<unsigned int> lower_face_dofs =
            surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<unsigned int> upper_face_dofs =
            surfaces[j]->get_dof_association_by_boundary_id(i);
        if (lower_face_dofs.size() != upper_face_dofs.size()) {
          std::cout << "ERROR: There was a edge dof count error!" << std::endl
              << "Surface " << i << " offers " << lower_face_dofs.size()
              << " dofs, " << j << " offers " << upper_face_dofs.size() << "."
              << std::endl;
        }
        for (unsigned int dof = 0; dof < lower_face_dofs.size(); dof++) {
          unsigned int dof_a = lower_face_dofs[dof] + surface_first_dofs[i];
          unsigned int dof_b = upper_face_dofs[dof] + surface_first_dofs[j];
          if (!constraints.is_constrained(dof_a)) {
            constraints.add_line(dof_a);
            constraints.add_entry(dof_a, dof_b, std::complex<double>(-1, 0));
          } else {
            if (!constraints.is_constrained(dof_b)) {
              constraints.add_line(dof_b);
              constraints.add_entry(dof_b, dof_a, std::complex<double>(-1, 0));
            } else {
              std::cout << "Both dofs already restrained ..." << std::endl;
            }
          }

          std::vector<unsigned int> current_line_a;
          std::vector<unsigned int> current_line_b;
          sp->add(dof_a, dof_b);
          for (auto temp_it = sp->begin(dof_b); temp_it != sp->end(dof_b);
              temp_it++) {
            current_line_a.push_back(temp_it->column());
          }
          sp->add(dof_b, dof_a);
          for (auto temp_it = sp->begin(dof_a); temp_it != sp->end(dof_a);
              temp_it++) {
            current_line_b.push_back(temp_it->column());
          }
          for (auto it_a = current_line_a.begin(); it_a != current_line_a.end();
              it_a++) {
            sp->add_entries(*it_a, current_line_b.begin(),
                current_line_b.end());
          }
          for (auto it_b = current_line_b.begin(); it_b != current_line_b.end();
              it_b++) {
            sp->add_entries(*it_b, current_line_a.begin(),
                current_line_a.end());
          }
        }
      }
    }
  }

  std::cout << "End Make Constraints." << std::endl;
}

void LocalProblem::assemble() {
  std::cout << "Start LocalProblem::assemble()" << std::endl;
  base_problem.assemble_system();
  sp = new dealii::SparsityPattern(n_own_dofs, n_own_dofs, 400);
  rhs.reinit(n_own_dofs);
  base_problem.make_sparsity_pattern(sp, 0);
  make_constraints();
  for (unsigned int surface = 0; surface < 6; surface++) {
    surfaces[surface]->fill_sparsity_pattern(sp, surface_first_dofs[surface]);
  }
  constraints.close();
  constraints.condense(*sp);
  sp->compress();
  matrix = new dealii::SparseMatrix<std::complex<double>>(*sp);

  std::cout << "Copy Main Matrix" << std::endl;
  for (auto it = base_problem.system_matrix.begin();
      it != base_problem.system_matrix.end(); it++) {
    matrix->add(it->row(), it->column(), it->value());
  }
  std::cout << "Done" << std::endl;
  for (unsigned int surface = 0; surface < 6; surface++) {
    std::cout << "Fill Surface Block " << surface << std::endl;
    surfaces[surface]->fill_matrix(matrix, surface_first_dofs[surface]);
  }
  std::cout << "Condense" << std::endl;

  constraints.condense(*matrix);
  constraints.distribute(rhs);
  std::cout << "Compress" << std::endl;
  matrix->compress(dealii::VectorOperation::add);
  std::cout << "End LocalProblem::assemble()" << std::endl;
  // validate();
}

void LocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
}

void LocalProblem::run() {
  std::cout << "Start LocalProblem::run()" << std::endl;
  assemble();
  solve();
  output_results();

  std::cout << "End LocalProblem::run()" << std::endl;
}

void LocalProblem::solve() {
  solver.solve(*matrix, rhs);
  std::cout << rhs.l2_norm() << std::endl;
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
    dealii::LinearAlgebra::distributed::Vector<std::complex<double>>) {

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

dealii::Vector<std::complex<double>> LocalProblem::get_local_vector_from_global() {
  dealii::Vector<std::complex<double>> ret(base_problem.dof_handler.n_dofs());
  std::cout << "Dof handler dofs: " << base_problem.dof_handler.n_dofs()
      << std::endl;
  for (unsigned int i = 0; i < base_problem.n_dofs; i++) {
    ret[i] = rhs[i];
  }
  return ret;
}

void LocalProblem::output_results() {
  dealii::DataOut<3> data_out;
  dealii::Vector<std::complex<double>> solution =
      get_local_vector_from_global();
  data_out.attach_dof_handler(base_problem.dof_handler);
  data_out.add_data_vector(solution, "Solution");
  data_out.build_patches();
  std::ofstream outputvtu("solution.vtu");
  data_out.write_vtu(outputvtu);

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
