#include "../Core/Types.h"
#include "LocalProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/vector.h>

LocalProblem::LocalProblem() :
    HierarchicalProblem(0), base_problem() {
  base_problem.make_grid();
}

LocalProblem::~LocalProblem() {}

void LocalProblem::solve(NumericVectorLocal src,
    NumericVectorLocal &dst) {
  NumericVectorLocal inputb(n_own_dofs);
  for (unsigned int i = 0; i < n_own_dofs; i++) {
    inputb[i] = src(i);
  }

  solver.solve(inputb);

  for (unsigned int i = 0; i < n_own_dofs; i++) {
    dst[i] = inputb[i];
  }
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
    surfaces[side] = std::shared_ptr<HSIESurface>(new HSIESurface(5, std::ref(surf_tria), side,
          GlobalParams.So_ElementOrder, k0, additional_coorindate));
    surfaces[side]->initialize();
    Position center = {0,0,0};
    surfaces[side]->set_V0(center);
  }

  std::cout << "Initialize index sets" << std::endl;
  initialize_own_dofs();
  std::cout << "Number of local dofs: " << n_own_dofs << std::endl;
}

void LocalProblem::generate_sparsity_pattern() {
}

void LocalProblem::validate() {
  std::cout << "Validate System: " << std::endl;
  std::cout << "N Rows: " << matrix->m() << std::endl;
  std::cout << "N Cols: " << matrix->n() << std::endl;
  std::cout << "Actually non-zero elements: "
      << matrix->n_actually_nonzero_elements(0.001) << std::endl;
  std::cout << "Matrix l1 norm: " << matrix->l1_norm() << std::endl;
  std::complex<double> matrix_entry;
  for (unsigned int i = 0; i < matrix->n(); i++) {
    auto it = matrix->begin(i);
    auto end = matrix->end(i);
    unsigned int entries = 0;
    while (it != end) {
      matrix_entry = (*it).value();
      if (std::abs(matrix_entry) != 0.0 ) {
        entries++;
      }
      it++;
    }
    if (entries == 0) {
      std::cout << "There were no entries in line " << i << std::endl;
    }

  }

}

DofCount LocalProblem::compute_own_dofs() {
  std::cout << "Begin Compute own dofs: " << std::endl;
  surface_first_dofs.clear();
  DofCount ret = base_problem.dof_handler.n_dofs();
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    ret += surfaces[i]->dof_counter;
    if (i != 5) {
      surface_first_dofs.push_back(ret);
    }
  }
  for (DofNumber i = 0; i < surface_first_dofs.size(); i++) {
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

  // couple surface dofs with inner ones.
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
  std::cout << "Restraints after phase 1:" << constraints.n_constraints()
      << std::endl;
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      surface_to_surface_constraints.reinit(is);
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
          surface_to_surface_constraints.add_line(dof_a);
          surface_to_surface_constraints.add_entry(dof_a, dof_b, std::complex<double>(-1, 0));
        }
      }
      constraints.merge(surface_to_surface_constraints, 
        dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins);
    }
  }
  std::cout << "Restraints after phase 2:" << constraints.n_constraints() << std::endl;

  std::cout << "End Make Constraints." << std::endl;
}

void LocalProblem::assemble() {
  std::cout << "Start LocalProblem::assemble()" << std::endl;
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

  std::cout << "Assemble Main Matrix" << std::endl;
  base_problem.assemble_system(0, matrix, &rhs);
  std::cout << "Done" << std::endl;
  for (unsigned int surface = 0; surface < 6; surface++) {
    std::cout << "Fill Surface Block " << surface << std::endl;
    surfaces[surface]->fill_matrix(matrix, surface_first_dofs[surface]);
  }
  std::cout << "Condense" << std::endl;

  constraints.condense(*matrix);
  // constraints.distribute(rhs);
  std::cout << "Compress" << std::endl;
  matrix->compress(dealii::VectorOperation::add);
  std::cout << "End LocalProblem::assemble()" << std::endl;
  validate();
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
  std::cout << "Solve the system." << std::endl;
  std::cout << "Norm before: " << rhs.l2_norm() << std::endl;
  solver.factorize(*matrix);

  solver.solve(rhs);

  constraints.distribute(rhs);
  std::cout << "Done solving." << std::endl;
  std::cout << "Norm after: " << rhs.l2_norm() << std::endl;
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
