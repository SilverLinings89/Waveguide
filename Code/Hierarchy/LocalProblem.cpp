#include "../Core/Types.h"
#include "LocalProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"
#include "../Helpers/staticfunctions.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include "../Helpers/PointSourceField.h"

PointSourceField psf;

LocalProblem::LocalProblem() :
    HierarchicalProblem(0), base_problem(), sc(), solver(sc, MPI_COMM_SELF) {
  base_problem.make_grid();
  MPI_Barrier(MPI_COMM_WORLD);
  print_info("Local Problem", "Done building base problem. Preparing matrix.");
  matrix = new dealii::PETScWrappers::SparseMatrix();
}

LocalProblem::~LocalProblem() {}

auto LocalProblem::get_center() -> Position const {
  return compute_center_of_triangulation(&base_problem.triangulation);
}

void LocalProblem::initialize() {
  print_info("LocalProblem::initialize", "Start");
  for (unsigned int side = 0; side < 6; side++) {
    dealii::Triangulation<2, 3> temp_triangulation;
    const unsigned int component = side / 2;
    double additional_coorindate = 0;
    bool found = false;
    for (auto it : base_problem.triangulation.active_cell_iterators()) {
      if (it->at_boundary(side)) {
        for (auto i = 0; i < 6 && !found; i++) {
          if (it->face(i)->boundary_id() == side) {
            found = true;
            additional_coorindate = it->face(i)->center()[component];
          }
        }
      }
      if (found) {
        break;
      }
    }
    dealii::Triangulation<2> surf_tria;
    print_info("LocalProblem::initialize", "Initializing surface " + std::to_string(side) + " in local problem.", false, LoggingLevel::PRODUCTION_ALL);
    Mesh tria;
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
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    surfaces[side] = std::shared_ptr<HSIESurface>(new HSIESurface(GlobalParams.HSIE_polynomial_degree, std::ref(surf_tria), side,
          GlobalParams.Nedelec_element_order, GlobalParams.kappa_0, additional_coorindate));
    surfaces[side]->initialize();
  }

  print_info("LocalProblem::initialize", "Initialize index sets", false, LoggingLevel::DEBUG_ALL);
  initialize_own_dofs();
  print_info("LocalProblem::initialize", "Number of local dofs: " + std::to_string(n_own_dofs) , false, LoggingLevel::DEBUG_ALL);
  print_info("LocalProblem::initialize", "End");
}

void LocalProblem::generate_sparsity_pattern() {
}

void LocalProblem::validate() {
  std::cout << "Validate System: " << std::endl;
  std::cout << "N Rows: " << matrix->m() << std::endl;
  std::cout << "N Cols: " << matrix->n() << std::endl;
  std::cout << "Matrix l1 norm: " << matrix->l1_norm() << std::endl;
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
    std::vector<DofIndexAndOrientationAndPosition> from_surface =
        surfaces[surface]->get_dof_association();
    std::vector<DofIndexAndOrientationAndPosition> from_inner_problem =
        base_problem.get_surface_dof_vector_for_boundary_id(surface);
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
  std::cout << "Constraints after phase 1:" << constraints.n_constraints()
      << std::endl;
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      surface_to_surface_constraints.reinit(is);
      bool opposing = ((i % 2) == 0) && (i + 1 == j);
      if (!opposing) {
        std::vector<DofIndexAndOrientationAndPosition> lower_face_dofs =
            surfaces[i]->get_dof_association_by_boundary_id(j);
        std::vector<DofIndexAndOrientationAndPosition> upper_face_dofs =
            surfaces[j]->get_dof_association_by_boundary_id(i);
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
  base_problem.make_constraints(&constraints, 0, own_dofs);
  std::cout << "Constraints after phase 3:" << constraints.n_constraints() << std::endl;
  std::cout << "End Make Constraints." << std::endl;
}

void LocalProblem::assemble() {
  std::cout << "Start LocalProblem::assemble()" << std::endl;
  base_problem.assemble_system(0, &constraints, matrix, &rhs);
  for (unsigned int surface = 0; surface < 6; surface++) {
    std::cout << "Fill Surface Block " << surface << std::endl;
    surfaces[surface]->fill_matrix(matrix, &rhs, surface_first_dofs[surface],get_center(), &constraints);
  }
  matrix->compress(dealii::VectorOperation::add);
  std::cout << "End LocalProblem::assemble()" << std::endl;
}

void LocalProblem::reinit() {
  dealii::DynamicSparsityPattern dsp = { n_own_dofs };
  rhs.reinit(MPI_COMM_SELF, n_own_dofs, n_own_dofs, false);
  solution.reinit(MPI_COMM_SELF, n_own_dofs, n_own_dofs, false);
  make_constraints();
  base_problem.make_sparsity_pattern(&dsp, 0, &constraints);
  for (unsigned int surface = 0; surface < 6; surface++) {
    surfaces[surface]->fill_sparsity_pattern(&dsp, surface_first_dofs[surface], &constraints);
  }
  constraints.close();
  sp.copy_from(dsp);
  matrix->reinit(sp);
}

void LocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
  own_dofs.set_size(n_own_dofs);
  own_dofs.add_range(0, n_own_dofs);
}

/**
void LocalProblem::run() {
  std::cout << "Start LocalProblem::run()" << std::endl;
  reinit();
  assemble();
  validate();
  solve();
  output_results();
  std::cout << "End LocalProblem::run()" << std::endl;
}
**/

void LocalProblem::solve() {
  std::cout << "Solve the system." << std::endl;
  std::cout << "Norm before: " << solution.l2_norm() << std::endl;
  
  rhs.compress(dealii::VectorOperation::add);
  constraints.set_zero(solution);
  Timer timer1, timer2;
  timer1.start ();
  solver.solve(*matrix, solution, rhs);
  timer1.stop();
  std::cout << "Elapsed CPU time: " << timer1.cpu_time() << " seconds." << std::endl;
  std::cout << "Elapsed walltime: " << timer1.wall_time() << " seconds." << std::endl;
  
  std::cout << "Norm after: " << solution.l2_norm() << std::endl;
  constraints.distribute(solution);
  Mat fact;
  KSPGetPC(solver.solver_data->ksp,&solver.solver_data->pc);
  PCFactorGetMatrix(solver.solver_data->pc,&fact);
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)),PETSC_VIEWER_ASCII_INFO);
  MatView(fact,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)));
  PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)));
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

dealii::Vector<ComplexNumber> LocalProblem::get_local_vector_from_global() {
  dealii::Vector<ComplexNumber> ret(base_problem.dof_handler.n_dofs());
  std::cout << "Dof handler dofs: " << base_problem.dof_handler.n_dofs()
      << std::endl;
  for (unsigned int i = 0; i < base_problem.n_dofs; i++) {
    ret[i] = solution[i];
  }
  return ret;
}

void LocalProblem::output_results() {
  dealii::DataOut<3> data_out;
  dealii::Vector<ComplexNumber> solution =
      get_local_vector_from_global();
  data_out.attach_dof_handler(base_problem.dof_handler);
  data_out.add_data_vector(solution, "Solution");
  std::ofstream outputvtu("solution.vtu");
  dealii::Vector<double> cellwise_error(base_problem.triangulation.n_active_cells());
  dealii::Vector<double> cellwise_norm(base_problem.triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    base_problem.dof_handler,
    solution,
    psf,
    cellwise_error,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  dealii::Vector<ComplexNumber> zero(base_problem.n_dofs);  
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    base_problem.dof_handler,
    zero,
    psf,
    cellwise_norm,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  unsigned int index = 0;
  for(auto it = base_problem.dof_handler.begin_active(); it != base_problem.dof_handler.end(); it++) {
    if(base_problem.constrained_cells.contains(it->id().to_string())) {
      cellwise_error[index] = 0;
      cellwise_norm[index] = 0;
    }
    index++;
  }
  const double global_error = dealii::VectorTools::compute_global_error(base_problem.triangulation, cellwise_error, dealii::VectorTools::NormType::L2_norm);
  const double global_norm = dealii::VectorTools::compute_global_error(base_problem.triangulation, cellwise_norm, dealii::VectorTools::NormType::L2_norm);
  std::cout << "Global computed error L2: " << global_error << std::endl;
  std::cout << "Exact solution L2 norm: " << global_norm << std::endl;
  data_out.add_data_vector(cellwise_error, "Cellwise_error");
  data_out.build_patches();
  data_out.write_vtu(outputvtu);
  compare_to_exact_solution();
}

auto LocalProblem::compare_to_exact_solution() -> void {
  NumericVectorLocal solution_inner(base_problem.n_dofs);
  for(unsigned int i = 0; i < base_problem.n_dofs; i++) {
    solution_inner[i] = solution[i];
  }

  psf.set_cell_diameter(GlobalParams.Geometry_Size_X / GlobalParams.Cells_in_x -0.0001);
  std::ofstream myfile ("output_z.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double z = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {0,0, z};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(base_problem.dof_handler, solution_inner, p, local_solution);
    psf.vector_value(p, exact_solution);
    myfile << "0\t0\t" << z << "\t" << local_solution[0].real()<< "\t" << local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
  myfile.open("output_y.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double y = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {0,y,0};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(base_problem.dof_handler, solution_inner, p, local_solution);
    psf.vector_value(p, exact_solution);
    myfile <<"0\t" << y << "\t0\t"<< local_solution[0].real()<< "\t"<< local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
  myfile.open("output_x.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double x = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {x,0,0};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(base_problem.dof_handler, solution_inner, p, local_solution);
    psf.vector_value(p, exact_solution);
    myfile << x << "\t0\t0";
    myfile << "\t" << local_solution[0].real()<< "\t"<< local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
}

auto LocalProblem::communicate_sweeping_direction(SweepingDirection sweeping_direction_of_parent) -> void {
  sweeping_direction = sweeping_direction_of_parent;
}

auto LocalProblem::set_boundary_values(dealii::IndexSet local_indices, std::vector<ComplexNumber> dof_values) -> void {
  if(local_indices.n_elements() == dof_values.size()) {
    std::vector<unsigned int> indices;
    for(auto item: local_indices) {
      indices.push_back(item);
    }
    rhs.set(indices, dof_values);
  } else {
    std::cout << "Boundary values were passed incorrectly.";
  }
}

auto LocalProblem::release_boundary_values(dealii::IndexSet local_indices) -> void {
  std::vector<unsigned int> indices;
  std::vector<ComplexNumber> values;
  for(auto item: local_indices) {
    indices.push_back(item);
    values.push_back(0);
  }
  rhs.set(indices, values);
}