#include "../Core/Types.h"
#include "LocalProblem.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "../BoundaryCondition/BoundaryCondition.h"
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

LocalProblem::LocalProblem() :
    HierarchicalProblem(0), base_problem(), sc(), solver(sc, MPI_COMM_SELF) {
  base_problem.make_grid();
  print_info("Local Problem", "Done building base problem. Preparing matrix.");
  matrix = new dealii::PETScWrappers::SparseMatrix();
  for(unsigned int i = 0; i < 6; i++) is_hsie_surface[i] = true;
  is_hsie_surface[4] = false;
  is_hsie_surface[5] = true;
}

LocalProblem::~LocalProblem() {}

auto LocalProblem::get_center() -> Position const {
  return compute_center_of_triangulation(&base_problem.triangulation);
}

dealii::IndexSet LocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  dealii::IndexSet ret(n_own_dofs);
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(interface_id);
      for(unsigned int j = 0; j < current.size(); j++) {
        ret.add_index(current[j].index + this->first_own_index);
      }      
    } else {
      if(i != opposing_interface_id && is_hsie_surface[i]) {
        std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->surfaces[i]->get_dof_association_by_boundary_id(i);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index + this->first_own_index);
        }
      }
    }
  }
  return ret;
}

void LocalProblem::initialize() {
  print_info("LocalProblem::initialize", "Start");
  for (unsigned int side = 0; side < 6; side++) {
    if(is_hsie_surface[side]) {
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
      print_info("LocalProblem::initialize", "Initializing surface " + std::to_string(side) + " in local problem.", false, LoggingLevel::DEBUG_ALL);
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
      if(GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE) {
        print_info("LocalProblem::initialize", "Using HSIE as Boundary Method");
        surfaces[side] = std::shared_ptr<BoundaryCondition>(new HSIESurface(GlobalParams.HSIE_polynomial_degree, std::ref(surf_tria), side, GlobalParams.Nedelec_element_order, GlobalParams.kappa_0, additional_coorindate));
      } else {
        print_info("LocalProblem::initialize", "Using PML as Boundary Method");
        surfaces[side] = std::shared_ptr<BoundaryCondition>(new PMLSurface(side, additional_coorindate, std::ref(surf_tria)));
      }
      surfaces[side]->initialize();
    } else {
      surfaces[side] = nullptr;
    }
  }
  print_info("LocalProblem::initialize", "Initialize index sets", false, LoggingLevel::DEBUG_ALL);
  initialize_own_dofs();
  print_info("LocalProblem::initialize", "Number of local dofs: " + std::to_string(n_own_dofs) , false, LoggingLevel::DEBUG_ALL);
  for(unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]){
      surface_dof_associations[i] = surfaces[i]->get_dof_association();
      for(unsigned int j = 0; j < surface_dof_associations[i].size(); j++) {
        surface_dof_index_vectors[i].push_back(first_own_index + surface_dof_associations[i][j].index);
      }
    }
  }
  for(unsigned int i = 0; i < 6; i++) {
    surface_index_sets[i] = compute_interface_dof_set(i);
  }
  if(GlobalParams.NumberProcesses == 1) {
    reinit();
  }
  print_info("LocalProblem::initialize", "End");
}

void LocalProblem::generate_sparsity_pattern() {
}

void LocalProblem::validate() {
  print_info("LocalProblem::validate", "Start");
  print_info("LocalProblem::validate", "Matrix size: (" + std::to_string(matrix->m()) + " x "  + std::to_string(matrix->n()) + ") and l1-norm " + std::to_string(matrix->l1_norm()), false, LoggingLevel::PRODUCTION_ONE);
  print_info("LocalProblem::validate", "End");
}

DofCount LocalProblem::compute_own_dofs() {
  print_info("LocalProblem::compute_own_dofs", "Start");
  surface_first_dofs.clear();
  DofCount ret = base_problem.dof_handler.n_dofs();
  surface_first_dofs.push_back(ret);
  for (unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]) {
      ret += surfaces[i]->dof_counter;
      if (i != 5) {
        surface_first_dofs.push_back(ret);
      }
    } else {
      if (i != 5) {
        surface_first_dofs.push_back(ret);
      }
    }
  }
  print_info("LocalProblem::compute_own_dofs", "End");
  return ret;
}

void LocalProblem::make_constraints() {
  print_info("LocalProblem::make_constraints", "Start");
  dealii::IndexSet is;
  is.set_size(n_own_dofs);
  is.add_range(0, n_own_dofs);
  constraints.reinit(is);

  // couple surface dofs with inner ones.
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]) {
      std::vector<DofIndexAndOrientationAndPosition> from_surface = surfaces[surface]->get_dof_association();
      std::vector<DofIndexAndOrientationAndPosition> from_inner_problem = base_problem.get_surface_dof_vector_for_boundary_id(surface);
      if (from_surface.size() != from_inner_problem.size()) {
        std::cout << "Warning: Size mismatch in make_constraints for surface "
            << surface << ": Inner: " << from_inner_problem.size()
            << " != Surface:" << from_surface.size() << "." << std::endl;
      }
      bool found_incompatible_surface_dof = false;
      for (unsigned int line = 0; line < from_inner_problem.size(); line++) {
        if (!areDofsClose(from_inner_problem[line], from_surface[line])) {
          found_incompatible_surface_dof = true;
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
      if(found_incompatible_surface_dof) std::cout << "There was atleast one incompatible dof for " << surface << std::endl;
    }
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 1: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );
  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]) surfaces[i]->setup_neighbor_couplings(is_hsie_surface);
  }
  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = i + 1; j < 6; j++) {
      if(is_hsie_surface[i] && is_hsie_surface[j]) {
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
          bool found_incompatible_dof = false;
          for (unsigned int dof = 0; dof < lower_face_dofs.size(); dof++) {
            if (!areDofsClose(lower_face_dofs[dof], upper_face_dofs[dof])) {
              found_incompatible_dof = true;
            }
            unsigned int dof_a = lower_face_dofs[dof].index + surface_first_dofs[i];
            unsigned int dof_b = upper_face_dofs[dof].index + surface_first_dofs[j];
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
          if(found_incompatible_dof) std::cout << "For surface " << i << " and " << j << ": Found incompatible dof." << std::endl;
        }
        constraints.merge(surface_to_surface_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins);
      }
    }
  }
  for (unsigned int i = 0; i < 6; i++) {
    if(is_hsie_surface[i]) surfaces[i]->reset_neighbor_couplings(is_hsie_surface);
  }
  print_info("LocalProblem::make_constraints", "Constraints after phase 2: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );

  base_problem.make_constraints(&constraints, 0, own_dofs);
  print_info("LocalProblem::make_constraints", "Constraints after phase 3: " + std::to_string(constraints.n_constraints()), false, LoggingLevel::DEBUG_ALL );

  print_info("LocalProblem::make_constraints", "End");
}

void LocalProblem::assemble() {
  base_problem.assemble_system(0, &constraints, matrix, &rhs);
  // 0 broken
  // 1 works
  // 2 broken
  // 3 works
  // 4 broken
  // 5 works
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]) {
      std::cout << "Surface " << surface << std::endl;
      surfaces[surface]->fill_matrix(matrix, &rhs, surface_first_dofs[surface], is_hsie_surface, &constraints);
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs.compress(dealii::VectorOperation::add);
}

void LocalProblem::reinit_rhs() {
  // rhs = new NumericVectorDistributed(MPI_COMM_SELF, n_own_dofs, n_own_dofs);
  // rhs = new dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[local_level]);
}

void LocalProblem::reinit() {
  dealii::DynamicSparsityPattern dsp = { n_own_dofs };
  reinit_rhs();
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, MPI_COMM_SELF);
  solution.reinit(MPI_COMM_SELF, n_own_dofs, n_own_dofs, false);
  make_constraints();
  base_problem.make_sparsity_pattern(&dsp, 0, &constraints);
  constraints.close();
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(is_hsie_surface[surface]) {
      std::cout << "Fill surface sparsity pattern for " << surface << std::endl;
      surfaces[surface]->fill_sparsity_pattern(&dsp, surface_first_dofs[surface], &constraints);
    }
  }
  sp.copy_from(dsp);
  matrix->reinit(sp);
}

void LocalProblem::initialize_own_dofs() {
  n_own_dofs = compute_own_dofs();
  own_dofs.set_size(n_own_dofs);
  own_dofs.add_range(0, n_own_dofs);
}

void LocalProblem::solve() {
  // print_info("LocalProblem::solve", "Start");
  // print_info("LocalProblem::solve", "Norm before: " + std::to_string(solution.l2_norm()), false, LoggingLevel::DEBUG_ONE);
  // constraints.set_zero(solution);
  Timer timer1;
  timer1.start ();
  // dealii::PETScWrappers::MPI::Vector temp_rhs = *rhs;
  constraints.set_zero(solution);
  solver.solve(*matrix, solution, rhs);
  constraints.distribute(solution);
  timer1.stop();
  // print_info("LocalProblem::solve", "Elapsed CPU time: " + std::to_string(timer1.cpu_time()) + " seconds.", false, LoggingLevel::DEBUG_ONE);
  // print_info("LocalProblem::solve", "Elapsed walltime: " + std::to_string(timer1.wall_time()) + " seconds.", false, LoggingLevel::DEBUG_ONE);
  // print_info("LocalProblem::solve", "Norm after: " + std::to_string(solution.l2_norm()) + " seconds.", false, LoggingLevel::DEBUG_ALL);
  // constraints.distribute(solution);
  // Mat fact;
  // KSPGetPC(solver.solver_data->ksp,&solver.solver_data->pc);
  // PCFactorGetMatrix(solver.solver_data->pc,&fact);
  // PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)),PETSC_VIEWER_ASCII_INFO);
  // MatView(fact,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)));
  // PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fact)));
  // print_info("LocalProblem::solve", "End");
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
  print_info("LocalProblem::get_local_vector_from_global", "Start");
  dealii::Vector<ComplexNumber> ret(base_problem.dof_handler.n_dofs());
  for (unsigned int i = 0; i < base_problem.n_dofs; i++) {
    ret[i] = solution(i);
  }
  print_info("LocalProblem::get_local_vector_from_global", "End");
  return ret;
}

void LocalProblem::output_results(std::string filename) {
  print_info("LocalProblem::output_results()", "Start");
  dealii::Vector<double> epsilon(base_problem.triangulation.n_active_cells()); 
  unsigned int cnt = 0;
  for(auto it = base_problem.triangulation.begin_active(); it != base_problem.triangulation.end(); it++) {
    epsilon[cnt] = (Geometry.math_coordinate_in_waveguide(it->center())) ? GlobalParams.Epsilon_R_in_waveguide : GlobalParams.Epsilon_R_outside_waveguide;
    cnt++;
  }
  dealii::DataOut<3> data_out;
  dealii::Vector<ComplexNumber> output_solution = get_local_vector_from_global();
  data_out.attach_dof_handler(base_problem.dof_handler);
  data_out.add_data_vector(epsilon,"Epsilon", dealii::DataOut_DoFData<DoFHandler<3, 3>, 3, 3>::type_cell_data);
  data_out.add_data_vector(output_solution, "Solution");
  std::ofstream outputvtu(filename + std::to_string(GlobalParams.MPI_Rank) + ".vtu");
  dealii::Vector<double> cellwise_error(base_problem.triangulation.n_active_cells());
  dealii::Vector<double> cellwise_norm(base_problem.triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    base_problem.dof_handler,
    output_solution,
    *GlobalParams.source_field,
    cellwise_error,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  dealii::Vector<ComplexNumber> zero(base_problem.n_dofs);  
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    base_problem.dof_handler,
    zero,
    *GlobalParams.source_field,
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
  print_info("LocalProblem::output_results", "Global computed error L2: " + std::to_string(global_error), false, LoggingLevel::PRODUCTION_ONE);
  print_info("LocalProblem::output_results", "Exact solution L2 norm: " + std::to_string(global_norm), false, LoggingLevel::PRODUCTION_ONE);
  data_out.add_data_vector(cellwise_error, "Cellwise_error");
  data_out.build_patches();
  data_out.write_vtu(outputvtu);
  write_phase_plot();
  print_info("LocalProblem::output_results()", "End");
}

auto LocalProblem::write_phase_plot() -> void {
  dealii::Vector<ComplexNumber> output_solution = get_local_vector_from_global();
  const unsigned int n_points = 50;
  std::ofstream outfile;
  outfile.open("Phase_Plot" + std::to_string(GlobalParams.MPI_Rank) + ".dat");
  for(unsigned int i = 0; i < n_points; i++) {
    dealii::Vector<ComplexNumber> numeric_solution(3);
    dealii::Vector<ComplexNumber> exact_solution(3);
    Point<3, double> location = Point<3,double>(0,0, Geometry.global_z_range.first + i * (Geometry.global_z_range.second - Geometry.global_z_range.first)/n_points);
    dealii::VectorTools::point_value(base_problem.dof_handler, output_solution, location , numeric_solution);
    GlobalParams.source_field->vector_value(location, exact_solution);
    outfile << location[2] << "\t";
    for(unsigned int j = 0; j < 3; j++) {
      outfile << exact_solution[j].real() << "\t"<< exact_solution[j].imag() << "\t";
    }
    for(unsigned int j = 0; j < 3; j++) {
      outfile << numeric_solution[j].real() << "\t"<< numeric_solution[j].imag() << "\t";
    }
    for(unsigned int j = 0; j < 3; j++) {
      outfile << numeric_solution[j].real() - exact_solution[j].real() << "\t"<< numeric_solution[j].imag()  - exact_solution[j].imag()<< "\t";
    }
    outfile << std::endl;
  }
}

auto LocalProblem::compare_to_exact_solution() -> void {
  NumericVectorLocal solution_inner(base_problem.n_dofs);
  for(unsigned int i = 0; i < base_problem.n_dofs; i++) {
    solution_inner[i] = solution(i);
  }

  std::ofstream myfile ("output_z.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double z = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {0,0, z};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(base_problem.dof_handler, solution_inner, p, local_solution);
    GlobalParams.source_field->vector_value(p, exact_solution);
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
    GlobalParams.source_field->vector_value(p, exact_solution);
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
    GlobalParams.source_field->vector_value(p, exact_solution);
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

void LocalProblem::update_mismatch_vector(BoundaryId in_bid) {
  rhs_mismatch.reinit( MPI_COMM_SELF, n_own_dofs, n_own_dofs);
  for(unsigned int i = 0; i < surfaces[in_bid]->dof_counter; i++) {
    solution[surface_first_dofs[in_bid] + i] = 0;
  }
  std::vector<DofIndexAndOrientationAndPosition> current = get_local_problem()->base_problem.get_surface_dof_vector_for_boundary_id(in_bid);
  for(unsigned int i = 0; i < current.size(); i++) {
    solution[current[i].index + first_own_index] = 0;
  }
  solution.compress(VectorOperation::insert);
  matrix->vmult(rhs_mismatch, solution);
  std::cout << "RHS Mismatch on is " << rhs_mismatch.l2_norm() << " for input norm " << solution.l2_norm() << std::endl; 
}

void LocalProblem::compute_solver_factorization() {
  Timer timer1;
  // print_info("LocalProblem::compute_solver_factorization", "Begin solver factorization: ", true, LoggingLevel::PRODUCTION_ONE);
  timer1.start();
  solve();
  timer1.stop();
  // print_info("LocalProblem::compute_solver_factorization", "Walltime: " + std::to_string(timer1.wall_time()) , true, LoggingLevel::PRODUCTION_ONE);
}

double LocalProblem::compute_L2_error() {
  NumericVectorLocal solution_inner(base_problem.n_dofs);
  for(unsigned int i = 0; i < base_problem.n_dofs; i++) {
    solution_inner[i] = solution(i);
  }
  dealii::Vector<double> cellwise_error(base_problem.triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    base_problem.dof_handler,
    solution_inner,
    *GlobalParams.source_field,
    cellwise_error,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  return dealii::VectorTools::compute_global_error(base_problem.triangulation, cellwise_error, dealii::VectorTools::NormType::L2_norm);
}