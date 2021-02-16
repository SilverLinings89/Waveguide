#include "RectangularMode.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include "../Helpers/staticfunctions.h"
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_solver.h>
// #include <deal.II/lac/slepc_solver.h>

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
  Position p1(-GlobalParams.Geometry_Size_X / 2.0, -GlobalParams.Geometry_Size_Y / 2.0, 0);
  Position p2(GlobalParams.Geometry_Size_X / 2.0, GlobalParams.Geometry_Size_Y / 2.0, layer_thickness);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(20);
  repetitions.push_back(20);
  repetitions.push_back(1);
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2, true);
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
  
  for (unsigned int side = 0; side < 4; side++) {      
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
    dealii::GridGenerator::extract_boundary_mesh(tria, temp_triangulation, b_ids);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    surfaces[side] = std::shared_ptr<HSIESurface>(new HSIESurface(10, std::ref(surf_tria), side, 0, GlobalParams.kappa_0, additional_coorindate));
    surfaces[side]->initialize();
  }
  n_dofs_total = dof_handler.n_dofs();
  for(unsigned int i = 0; i < 4; i++) {
    n_dofs_total += surfaces[i]->dof_counter;
  }
  dealii::IndexSet is(n_dofs_total);
  constraints.reinit(is);
  is.add_range(0,n_dofs_total);
  periodic_constraints.reinit(is);
  ComplexNumber factor = std::exp(layer_thickness / (lambda/std::sqrt(GlobalParams.Epsilon_R_in_waveguide)) * 2*GlobalParams.Pi * ComplexNumber(0.0,1.0)); 
  DoFTools::make_periodicity_constraints(dof_handler, 4, 5, 2, periodic_constraints, dealii::ComponentMask() ,factor);
  std::cout << "N Constraints: " << periodic_constraints.n_constraints() << std::endl;
  surface_first_dofs.clear();
  DofCount dc = dof_handler.n_dofs();
  surface_first_dofs.push_back(dc);
  for (unsigned int i = 0; i < 4; i++) {
    dc += surfaces[i]->dof_counter;
    if (i != 3) {
      surface_first_dofs.push_back(dc);
    }
  }

  constraints.merge(periodic_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::no_conflicts_allowed, true);
  for(unsigned int surf_index = 0; surf_index < 4; surf_index++) {
    std::vector<InterfaceDofData> side_4 = surfaces[surf_index]->get_dof_association_by_boundary_id(4);
    std::vector<InterfaceDofData> side_5 = surfaces[surf_index]->get_dof_association_by_boundary_id(5);
    for(unsigned int i = 0; i < side_4.size(); i++) {
      constraints.add_line(side_4[i].index+surface_first_dofs[surf_index]);
      constraints.add_entry(side_4[i].index+surface_first_dofs[surf_index], side_5[i].index + surface_first_dofs[surf_index], factor);
    }
  }
  for (unsigned int surface = 0; surface < 4; surface++) {
    std::vector<InterfaceDofData> from_surface = surfaces[surface]->get_dof_association();
    std::vector<InterfaceDofData> from_inner_problem = get_surface_dof_vector_for_boundary_id(surface);
    if (from_surface.size() != from_inner_problem.size()) {
      std::cout << "Warning: Size mismatch in make_constraints for surface "
          << surface << ": Inner: " << from_inner_problem.size()
          << " != Surface:" << from_surface.size() << "." << std::endl;
    }
    shift_interface_dof_data(&from_surface, surface_first_dofs[surface]);
    AffineConstraints<ComplexNumber> temp_constraints;
    for (unsigned int line = 0; line < from_inner_problem.size(); line++) {
      if (!areDofsClose(from_inner_problem[line], from_surface[line])) {
        std::cout << "Error in face to inner_coupling. Positions are inner: "
            << from_inner_problem[line].base_point << " and surface: "
            << from_surface[line].base_point << std::endl;
      }
      temp_constraints = get_affine_constraints_for_InterfaceData(from_surface, from_inner_problem, n_dofs_total);
      constraints.merge(temp_constraints, AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins, true);
      temp_constraints.clear();
    }
  }

  dealii::AffineConstraints<ComplexNumber> surface_to_surface_constraints;
  for (unsigned int i = 0; i < 4; i++) {
    for (unsigned int j = i + 1; j < 4; j++) {
      surface_to_surface_constraints.reinit(is);
      bool opposing = ((i % 2) == 0) && (i + 1 == j);
      if (!opposing) {
        std::vector<InterfaceDofData> lower_face_dofs = surfaces[i]->get_dof_association_by_boundary_id(j);
        shift_interface_dof_data(&lower_face_dofs, surface_first_dofs[i]);
        std::vector<InterfaceDofData> upper_face_dofs = surfaces[j]->get_dof_association_by_boundary_id(i);
        shift_interface_dof_data(&upper_face_dofs, surface_first_dofs[j]);
        if (lower_face_dofs.size() != upper_face_dofs.size()) {
          std::cout << "ERROR: There was a edge dof count error!" << std::endl
              << "Surface " << i << " offers " << lower_face_dofs.size()
              << " dofs, " << j << " offers " << upper_face_dofs.size() << "."
              << std::endl;
        }
        surface_to_surface_constraints = get_affine_constraints_for_InterfaceData(lower_face_dofs, upper_face_dofs, n_dofs_total); 
        constraints.merge(surface_to_surface_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins);
        surface_to_surface_constraints.clear();
      }
    }
  }
  
  /**
  auto it = dealii::GridTools::find_active_cell_around_point(dof_handler, Position(GlobalParams.Width_of_waveguide/2.0,GlobalParams.Height_of_waveguide/2.0,0));
  std::vector<DofNumber> local_indices(fe.dofs_per_cell);
  it->get_dof_indices(local_indices);
  unsigned int lowest_dof = local_indices[0];
  print_info("Set value: ", lowest_dof, false, LoggingLevel::PRODUCTION_ONE);
  constraints.add_line(lowest_dof);
  constraints.set_inhomogeneity(lowest_dof, ComplexNumber(1.0, 0.0));
  **/
  constraints.close();
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
  dealii::Vector<ComplexNumber> base_vector(dof_handler.n_dofs());
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
                base_vector = 0;
                base_vector[line_dofs[i]] = 1;
                NumericVectorLocal field_value(3);
                dealii::VectorTools::point_value(dof_handler, base_vector, new_item.base_point, field_value);
                new_item.shape_val_at_base_point = deal_vector_to_position(field_value);
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
            base_vector = 0;
            base_vector[item] = 1;
            NumericVectorLocal field_value(3);
            dealii::VectorTools::point_value(dof_handler, base_vector, new_item.base_point, field_value);
            new_item.shape_val_at_base_point = deal_vector_to_position(field_value);
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
    surfaces[surface]->fill_sparsity_pattern(&dsp, surface_first_dofs[surface], &constraints);
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
      constraints.distribute_local_to_global(cell_data.cell_mass_matrix, cell_data.local_dof_indices, mass_matrix);
      constraints.distribute_local_to_global(cell_data.cell_stiffness_matrix, cell_data.local_dof_indices, stiffness_matrix);
    }
  }
  std::array<bool, 6> is_hsie = {true, true, true, true, false, false};
  for(unsigned int surf = 0; surf < 4; surf++) {
    surfaces[surf]->fill_matrix(&mass_matrix, &stiffness_matrix, &rhs, surface_first_dofs[surf], is_hsie, &constraints);
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

