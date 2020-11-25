#include "RectangularMode.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include "../Helpers/staticfunctions.h"
#include <deal.II/lac/slepc_solver.h>

using namespace dealii;

RectangularMode::RectangularMode(double x_width_waveguide,
                                 double y_width_waveguide,
                                 double x_width_domain, double y_width_domain,
                                 double beta, unsigned int order)
    : FEM_order(order),
      fe(order),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
      dof_handler(triangulation)
{
  this->x_width_domain = x_width_domain;
  this->y_width_domain = y_width_domain;
  this->x_width_waveguide = x_width_waveguide;
  this->y_width_waveguide = y_width_waveguide;
  this->beta = beta;
}

void RectangularMode::run() {
  make_mesh();
  make_boundary_conditions();
  assemble_system();
  output_solution();
}

void RectangularMode::make_mesh() {
  Position p1(-x_width_domain / 2.0, -y_width_domain / 2.0, -0.1);
  Position p2(x_width_domain / 2.0, y_width_domain / 2.0, 0.1);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(100);
  repetitions.push_back(100);
  repetitions.push_back(1);
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2, true);
  dof_handler.distribute_dofs(fe);
}

void RectangularMode::make_boundary_conditions() {
  DoFTools::make_periodicity_constraints(dof_handler, 4, 5, 2,
                                         periodic_constraints);
  for (unsigned int side = 0; side < 3; side++) {      
    dealii::Triangulation<2, 3> temp_triangulation;
    const unsigned int component = side / 2;
    double additional_coorindate = 0;
    bool found = false;
    for (auto it : triangulation.active_cell_iterators()) {
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
    Mesh tria;
    tria.copy_triangulation(triangulation);
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
    surfaces[side] = std::shared_ptr<HSIESurface>(new HSIESurface(10, std::ref(surf_tria), side, 0, GlobalParams.kappa_0, additional_coorindate));
    surfaces[side]->initialize();
  }

  surface_first_dofs.clear();
  DofCount dc = dof_handler.n_dofs();
  surface_first_dofs.push_back(dc);
  for (unsigned int i = 0; i < 4; i++) {
    dc += surfaces[i]->dof_counter;
    if (i != 5) {
      surface_first_dofs.push_back(dc);
    }
  }

  constraints.merge(periodic_constraints);

  for (unsigned int surface = 0; surface < 4; surface++) {
    std::vector<DofIndexAndOrientationAndPosition> from_surface = surfaces[surface]->get_dof_association();
    std::vector<DofIndexAndOrientationAndPosition> from_inner_problem = get_surface_dof_vector_for_boundary_id(surface);
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
      constraints.add_entry(from_inner_problem[line].index, from_surface[line].index + surface_first_dofs[surface], value);
    }
  }

  constraints.close();
}

std::vector<DofIndexAndOrientationAndPosition> RectangularMode::get_surface_dof_vector_for_boundary_id(
    unsigned int b_id) {
  std::vector<DofIndexAndOrientationAndPosition> ret;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe.dofs_per_face);
  std::set<DofNumber> face_set;
  triangulation.clear_user_flags();
  for (auto cell : dof_handler.active_cell_iterators()) {
    if (cell->at_boundary(b_id)) {
      bool found_one = false;
      for (unsigned int face = 0; face < 6; face++) {
        if (cell->face(face)->boundary_id() == b_id && found_one) {
          print_info("NumericProblem::get_surface_dof_vector_for_boundary_id", "There was an error!", false, LoggingLevel::PRODUCTION_ALL);
        }
        if (cell->face(face)->boundary_id() == b_id) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<DofIndexAndOrientationAndPosition> cell_dofs_and_orientations_and_points;
          for (unsigned int i = 0; i < dealii::GeometryInfo<3>::lines_per_face; i++) {
            std::vector<DofNumber> line_dofs(fe.dofs_per_line);
            cell->face(face)->line(i)->get_dof_indices(line_dofs);
            line_set.clear();
            line_set.insert(line_dofs.begin(), line_dofs.end());
            for(auto erase_it: line_set) {
              face_set.erase(erase_it);
            }
            if(!cell->face(face)->line(i)->user_flag_set()) {
              for (unsigned int j = 0; j < fe.dofs_per_line; j++) {
                DofIndexAndOrientationAndPosition new_item;
                new_item.index = line_dofs[j];
                new_item.position = cell->face(face)->line(i)->center();
                new_item.orientation = get_orientation(cell->face(face)->line(i)->vertex(0),
                            cell->face(face)->line(i)->vertex(1));
                cell_dofs_and_orientations_and_points.emplace_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          for (auto item: face_set) {
            DofIndexAndOrientationAndPosition new_item;

            new_item.index = item;
            new_item.position = cell->face(face)->center();
            new_item.orientation = get_orientation(cell->face(face)->vertex(0),
                    cell->face(face)->vertex(1));
            cell_dofs_and_orientations_and_points.emplace_back(new_item);
          }
          for (auto item: cell_dofs_and_orientations_and_points) {
            ret.push_back(item);
          }
        }
      }
    }
  }
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  return ret;
}

struct CellwiseAssemblyData {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Position> quadrature_points;
  const unsigned int dofs_per_cell;
  const unsigned int n_q_points;
  FullMatrix<ComplexNumber> cell_mass_matrix;
  FullMatrix<ComplexNumber> cell_stiffness_matrix;
  const double eps_in;
  const double eps_out;
  const double mu_zero;
  Vector<ComplexNumber> cell_rhs;
  MaterialTensor transformation;
  MaterialTensor epsilon;
  MaterialTensor mu;
  std::vector<DofNumber> local_dof_indices;
  DofHandler3D::active_cell_iterator cell;
  DofHandler3D::active_cell_iterator end_cell;
  const Position bounded_cell;
  const FEValuesExtractors::Vector fe_field;
  CellwiseAssemblyData(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
  fe_values(*fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_mass_matrix(dofs_per_cell,dofs_per_cell),
  cell_stiffness_matrix(dofs_per_cell,dofs_per_cell),
  eps_in(GlobalParams.Epsilon_R_in_waveguide),
  eps_out(GlobalParams.Epsilon_R_outside_waveguide),
  mu_zero(1.0),
  cell_rhs(dofs_per_cell),
  local_dof_indices(dofs_per_cell),
  bounded_cell(0.0,0.0,0.0),
  fe_field(0)
  {
    cell_rhs = 0;
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        if (i == j) {
          transformation[i][j] = ComplexNumber(1, 0);
        } else {
          transformation[i][j] = ComplexNumber(0, 0);
        }
      }
    }
    cell = dof_handler->begin_active();
    end_cell = dof_handler->end();
  };

  void prepare_for_current_q_index(unsigned int q_index) {
    mu = invert(transformation);

    if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
      epsilon = transformation * eps_in;
    } else {
      epsilon = transformation * eps_out;
    }

    const double JxW = fe_values.JxW(q_index);
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
      Tensor<1, 3, ComplexNumber> I_Curl;
      Tensor<1, 3, ComplexNumber> I_Val;
      I_Curl = fe_values[fe_field].curl(i, q_index);
      I_Val = fe_values[fe_field].value(i, q_index);

      for (unsigned int j = 0; j < dofs_per_cell; j++) {
        Tensor<1, 3, ComplexNumber> J_Curl;
        Tensor<1, 3, ComplexNumber> J_Val;
        J_Curl = fe_values[fe_field].curl(j, q_index);
        J_Val = fe_values[fe_field].value(j, q_index);
        cell_stiffness_matrix[i][j] += I_Curl * Conjugate_Vector(J_Curl)* JxW;
        cell_mass_matrix[i][j] += - ( I_Val * Conjugate_Vector(J_Val)) * JxW;
      }
    }
  }

  Tensor<1, 3, ComplexNumber> Conjugate_Vector(
      Tensor<1, 3, ComplexNumber> input) {
    Tensor<1, 3, ComplexNumber> ret;

    for (int i = 0; i < 3; i++) {
      ret[i].real(input[i].real());
      ret[i].imag(-input[i].imag());
    }
    return ret;
  }

};

void RectangularMode::assemble_system() {
  n_dofs_total = dof_handler.n_dofs();
  for(unsigned int i = 0; i < 4; i++) {
    n_dofs_total += surfaces[i]->dof_counter;
  }

  rhs.reinit(MPI_COMM_SELF, n_dofs_total, n_dofs_total, false);
  DynamicSparsityPattern dsp(n_dofs_total, n_dofs_total);
  auto end = dof_handler.end();
  std::vector<DofNumber> cell_dof_indices(fe.dofs_per_cell);
  for(auto cell = dof_handler.begin_active(); cell != end; cell++) {
    cell->get_dof_indices(cell_dof_indices);
    constraints.add_entries_local_to_global(cell_dof_indices, dsp);
  }
  for (unsigned int surface = 0; surface < 4; surface++) {
    surfaces[surface]->fill_sparsity_pattern(&dsp, surface_first_dofs[surface], &constraints);
  }
  constraints.close();
  sp.copy_from(dsp);
  mass_matrix.reinit(sp);
  stiffness_matrix.reinit(sp);

  CellwiseAssemblyData cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
    IndexSet input_dofs_local_set(fe.dofs_per_cell);
    std::vector<Position> input_dof_centers(fe.dofs_per_cell);
    std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);
    cell_data.cell_mass_matrix = 0;
    cell_data.cell_stiffness_matrix = 0;
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
      constraints.distribute_local_to_global(cell_data.cell_mass_matrix, cell_data.local_dof_indices, mass_matrix);
      constraints.distribute_local_to_global(cell_data.cell_stiffness_matrix, cell_data.local_dof_indices, stiffness_matrix);
    }
  }
  mass_matrix.compress(dealii::VectorOperation::add);
  stiffness_matrix.compress(dealii::VectorOperation::add);

  for(unsigned int surf = 0; surf < 4; surf++) {
    surfaces[surf]->fill_matrix(&mass_matrix, &stiffness_matrix, &rhs, surface_first_dofs[surf], Position(0,0,0), &constraints);
  }

  SolverControl                    solver_control(dof_handler.n_dofs(), 1e-9);
  SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control);
  std::vector<ComplexNumber>                     eigenvalues;
  std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
  IndexSet own_dofs(n_dofs_total);
  own_dofs.add_range(0, n_dofs_total);
  for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
    eigenfunctions[i].reinit(own_dofs, MPI_COMM_SELF);
  eigenfunctions.resize(10);
  eigenvalues.resize(10);
  eigensolver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);
  eigensolver.set_problem_type(EPS_GHEP);
  eigensolver.solve(stiffness_matrix,
                    mass_matrix,
                    eigenvalues,
                    eigenfunctions,
                    eigenfunctions.size());
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  for(unsigned int i = 0; i < 10; i++){
    dealii::Vector<ComplexNumber> temp (dof_handler.n_dofs());
    for(unsigned int element = 0; element < dof_handler.n_dofs(); element++) {
      temp[element] = eigenfunctions[i][element];
    }
    data_out.add_data_vector(temp, std::string("eigenfunction_") + std::to_string(i));
  }
  data_out.build_patches();
  std::ofstream output("eigenvalues.vtu");
  data_out.write_vtu(output);
}

void RectangularMode::output_solution(){

}

