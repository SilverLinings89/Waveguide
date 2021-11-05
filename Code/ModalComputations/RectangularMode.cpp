#include "RectangularMode.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include "../Helpers/staticfunctions.h"
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_solver.h>
#include "../GlobalObjects/GlobalObjects.h"

using namespace dealii;

RectangularMode::RectangularMode()
    : fe(GlobalParams.Nedelec_element_order),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
      dof_handler(triangulation),
      layer_thickness(0.05),
      lambda(1.55)
{
}

void RectangularMode::run() {
  make_mesh();
  SortDofsDownstream();
  make_boundary_conditions();
  assemble_system();
  solve();
  output_solution();
}

void RectangularMode::make_mesh() {
  Triangulation<3> temp_tria;
  Position p1(-GlobalParams.Geometry_Size_X / 2.0, -GlobalParams.Geometry_Size_Y / 2.0, 0);
  Position p2(GlobalParams.Geometry_Size_X / 2.0, GlobalParams.Geometry_Size_Y / 2.0, layer_thickness);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(20);
  repetitions.push_back(20);
  repetitions.push_back(1);
  GridGenerator::subdivided_hyper_rectangle(temp_tria, repetitions, p1, p2, true);
  triangulation = reforge_triangulation(&temp_tria);
  dof_handler.distribute_dofs(fe);
}

void RectangularMode::SortDofsDownstream() {
  print_info("RectangularProblem::SortDofsDownstream", "Start");
  triangulation.clear_user_flags();
  std::vector<std::pair<DofNumber, Position>> current;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe.dofs_per_face);
  std::set<DofNumber> face_set;
  std::vector<DofNumber> local_cell_dofs(fe.dofs_per_cell);
  std::set<DofNumber> cell_set;
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_cell_dofs);
    cell_set.clear();
    cell_set.insert(local_cell_dofs.begin(), local_cell_dofs.end());
    for(unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++) {
      cell->face(face)->get_dof_indices(local_face_dofs);
      face_set.clear();
      face_set.insert(local_face_dofs.begin(), local_face_dofs.end());
      for(auto firstit : face_set) {
        cell_set.erase(firstit);
      }
      for(unsigned int line = 0; line < GeometryInfo<3>::lines_per_face; line++) {
        cell->face(face)->line(line)->get_dof_indices(local_line_dofs);
        line_set.clear();
        line_set.insert(local_line_dofs.begin(), local_line_dofs.end());
        for(auto firstit : line_set) {
          face_set.erase(firstit);
        }
        if(!cell->face(face)->line(line)->user_flag_set()){
          for(auto dof: line_set) {
            current.emplace_back(dof, cell->face(face)->line(line)->center());
          }
          cell->face(face)->line(line)->set_user_flag();
        }
      }
      if(!cell->face(face)->user_flag_set()){
        for(auto dof: face_set) {
          current.emplace_back(dof, cell->face(face)->center());
        }
        cell->face(face)->set_user_flag();
      }
    }
    for(auto dof: cell_set) {
      current.emplace_back(dof, cell->center());
    }
  }
  std::sort(current.begin(), current.end(), compareDofBaseData);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  dof_handler.renumber_dofs(new_numbering);
  print_info("RectangularProblem::SortDofsDownstream", "End");
}

void RectangularMode::make_boundary_conditions() {
  print_info("RectangularProblem::make_boundary_conditions", "Start");
  
  
  print_info("RectangularProblem::make_boundary_conditions", "End");
}

std::vector<InterfaceDofData> RectangularMode::get_surface_dof_vector_for_boundary_id(
    unsigned int b_id) {
  std::vector<InterfaceDofData> ret;
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
          print_info("InnerDomain::get_surface_dof_vector_for_boundary_id", "There was an error!", false, LoggingLevel::PRODUCTION_ALL);
        }
        if (cell->face(face)->boundary_id() == b_id) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<InterfaceDofData> cell_dofs_and_orientations_and_points;
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
                InterfaceDofData new_item;
                new_item.index = line_dofs[j];
                new_item.base_point = cell->face(face)->line(i)->center();
                cell_dofs_and_orientations_and_points.emplace_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          unsigned int order = 0;
          for (auto item: face_set) {
            InterfaceDofData new_item;
            new_item.order = order;
            new_item.index = item;
            new_item.base_point = cell->face(face)->center();
            cell_dofs_and_orientations_and_points.emplace_back(new_item);
            order++;
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
  dealii::Vector<ComplexNumber> cell_rhs;
  const double eps_in;
  const double eps_out;
  const double mu_zero;
  MaterialTensor transformation;
  MaterialTensor epsilon;
  MaterialTensor mu;
  std::vector<DofNumber> local_dof_indices;
  DofHandler3D::active_cell_iterator cell;
  DofHandler3D::active_cell_iterator end_cell;
  const FEValuesExtractors::Vector fe_field;
  CellwiseAssemblyData(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
  fe_values(*fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_mass_matrix(dofs_per_cell,dofs_per_cell),
  cell_stiffness_matrix(dofs_per_cell,dofs_per_cell),
  cell_rhs(dofs_per_cell),
  eps_in(GlobalParams.Epsilon_R_in_waveguide),
  eps_out(GlobalParams.Epsilon_R_outside_waveguide),
  mu_zero(1.0),
  local_dof_indices(dofs_per_cell),
  fe_field(0)
  {
    cell = dof_handler->begin_active();
    end_cell = dof_handler->end();
  };

  void prepare_for_current_q_index(unsigned int q_index) {
    cell_rhs = 0;
    double eps_kappa_2 = 4 * GlobalParams.Pi * GlobalParams.Pi / (GlobalParams.Lambda*GlobalParams.Lambda);
    double eps = RectangularMode::compute_epsilon_for_Position(quadrature_points[q_index]);
    eps_kappa_2 *= eps;
    epsilon = transformation * eps;
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
        cell_mass_matrix[i][j] += ( I_Val * Conjugate_Vector(J_Val)) * JxW * eps_kappa_2;
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
  print_info("RectangularProblem::assemble_system", "Start");
  std::cout << "Epsilon: " << GlobalParams.Epsilon_R_in_waveguide << std::endl;
  rhs.reinit(MPI_COMM_SELF, n_dofs_total, n_dofs_total, false);
  DynamicSparsityPattern dsp(n_dofs_total, n_dofs_total);
  auto end = dof_handler.end();
  std::vector<DofNumber> cell_dof_indices(fe.dofs_per_cell);
  for(auto cell = dof_handler.begin_active(); cell != end; cell++) {
    cell->get_dof_indices(cell_dof_indices);
    constraints.add_entries_local_to_global(cell_dof_indices, dsp);
  }
  for (unsigned int surface = 0; surface < 4; surface++) {
    surfaces[surface]->fill_sparsity_pattern(&dsp, &constraints);
  }
  constraints.close();
  sp.copy_from(dsp);
  mass_matrix.reinit(sp);
  stiffness_matrix.reinit(sp);

  CellwiseAssemblyData cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    cell_data.cell_mass_matrix = 0;
    cell_data.cell_stiffness_matrix = 0;
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
    }
    constraints.distribute_local_to_global(cell_data.cell_mass_matrix, cell_data.local_dof_indices, mass_matrix);
    constraints.distribute_local_to_global(cell_data.cell_stiffness_matrix, cell_data.local_dof_indices, stiffness_matrix);
  }
  for(unsigned int surf = 0; surf < 4; surf++) {
    surfaces[surf]->fill_matrix(&mass_matrix, &stiffness_matrix, &rhs, &constraints);
  }
  mass_matrix.compress(dealii::VectorOperation::add);
  stiffness_matrix.compress(dealii::VectorOperation::add);
  rhs.compress(dealii::VectorOperation::add);
  print_info("RectangularProblem::make_boundary_conditions", "End");
}

void RectangularMode::solve() {
  print_info("RectangularProblem::solve", "Start");
  dealii::SolverControl                    solver_control(n_dofs_total, 1e-6);
  // dealii::SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control);
  IndexSet own_dofs(n_dofs_total);
  own_dofs.add_range(0, n_dofs_total);
  eigenfunctions.resize(n_eigenfunctions);
  for (unsigned int i = 0; i < n_eigenfunctions; ++i)
    eigenfunctions[i].reinit(own_dofs, MPI_COMM_SELF);
  eigenvalues.resize(n_eigenfunctions);
  // eigensolver.set_which_eigenpairs(EPS_SMALLEST_MAGNITUDE);
  // eigensolver.set_problem_type(EPS_GNHEP);
  print_info("RectangularProblem::solve", "Starting solution for a system with " + std::to_string(n_dofs_total) + " degrees of freedom.");
  /**
  eigensolver.solve(stiffness_matrix,
                    mass_matrix,
                    eigenvalues,
                    eigenfunctions,
                    n_eigenfunctions);
                    **/
  for(unsigned int i =0 ; i < n_eigenfunctions; i++) {
    // constraints.distribute(eigenfunctions[0]);
    eigenfunctions[i] /= eigenfunctions[i].linfty_norm();
  }
  print_info("RectangularProblem::solve", "End");
}

double RectangularMode::compute_epsilon_for_Position(Position in_p) {
  double ret = 0; 
  if (std::abs(in_p[0]) < GlobalParams.Width_of_waveguide/2.0 && std::abs(in_p[1]) < GlobalParams.Height_of_waveguide/2.0 ) {
    ret = GlobalParams.Epsilon_R_in_waveguide;
  } else {
    ret = GlobalParams.Epsilon_R_outside_waveguide;
  }
  return ret;
}

void RectangularMode::output_solution(){
  dealii::Vector<double> epsilon(triangulation.n_active_cells()); 
  unsigned int cnt = 0;
  for(auto it = triangulation.begin_active(); it != triangulation.end(); it++) {
    epsilon[cnt] = compute_epsilon_for_Position(it->center());
    cnt++;
  }
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<dealii::Vector<ComplexNumber>> eigenfunctions_small;
  eigenfunctions_small.resize(n_eigenfunctions);
  data_out.add_data_vector(epsilon,"Epsilon", dealii::DataOut_DoFData<DoFHandler<3, 3>, 3, 3>::type_cell_data);
  for(unsigned int i = 0; i < n_eigenfunctions; i++){
    eigenfunctions_small[i].reinit(dof_handler.n_dofs());
    for(unsigned int element = 0; element < dof_handler.n_dofs(); element++) {
      eigenfunctions_small[i][element] = eigenfunctions[i][element];
    }
    data_out.add_data_vector(eigenfunctions_small[i], std::string("eigenfunction_") + std::to_string(i));
  }
  data_out.build_patches();
  std::ofstream output("eigenvalues.vtu");
  data_out.write_vtu(output);
}

