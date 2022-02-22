#include "WaveguideTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

WaveguideTransformation::WaveguideTransformation()
    : SpaceTransformation(),
      waveguide_width(Geometry.global_z_range.first, Geometry.global_z_range.second, GlobalParams.Number_of_sectors, true, true, true, true),
      waveguide_height(Geometry.global_z_range.first, Geometry.global_z_range.second, GlobalParams.Number_of_sectors, true, true, true, true),
      vertical_shift(Geometry.global_z_range.first, Geometry.global_z_range.second, GlobalParams.Number_of_sectors, true, true, true, true)      
       {
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      if(i == j) {
        I[i][j] = 1;
      } else {
        I[i][j] = 0;
      }
    }
  }
  
}

WaveguideTransformation::~WaveguideTransformation() {}

Position WaveguideTransformation::math_to_phys(Position coord) const {
  Position ret;
  if(GlobalParams.keep_waveguide_width_constant) {
    ret[0] = coord[0];
  } else {
    ret[0] = coord[0] * waveguide_width.evaluate_at(coord[2]);
  }
  if(GlobalParams.keep_waveguide_height_constant) {
    ret[1] = coord[1] + vertical_shift.evaluate_at(coord[2]);
  } else {
    ret[1] = (coord[1] + vertical_shift.evaluate_at(coord[2])) * waveguide_height.evaluate_at(coord[2]);
  }
  ret[2] = coord[2];
  return ret;
}

Position WaveguideTransformation::phys_to_math(Position coord) const {
  Position ret;
  if(GlobalParams.keep_waveguide_width_constant) {
    ret[0] = coord[0];
  } else {
    ret[0] = coord[0] / waveguide_width.evaluate_at(coord[2]);
  }
  if(GlobalParams.keep_waveguide_height_constant) {
    ret[1] = coord[1] - vertical_shift.evaluate_at(coord[2]);
  } else {
    ret[1] = (coord[1] / waveguide_height.evaluate_at(coord[2])) - vertical_shift.evaluate_at(coord[2]);
  }
  ret[2] = coord[2];
  return ret;
}

Tensor<2, 3, ComplexNumber>
WaveguideTransformation::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}

double WaveguideTransformation::get_dof(int index) const {
  std::pair<ResponsibleComponent, unsigned int> comp = map_dof_index(index);
  switch (comp.first)
  {
    case VerticalDisplacementComponent:
      return vertical_shift.get_dof_value(comp.second);
      break;
    case WaveguideHeightComponent:
      return waveguide_height.get_dof_value(comp.second);
      break;
    case WaveguideWidthComponent:
      return waveguide_width.get_dof_value(comp.second);
      break;
    default:
      break;
  }
  return 0.0;
}

double WaveguideTransformation::get_free_dof(int index) const {
  std::pair<ResponsibleComponent, unsigned int> comp = map_free_dof_index(index);
  switch (comp.first)
  {
    case VerticalDisplacementComponent:
      return vertical_shift.get_free_dof_value(comp.second);
      break;
    case WaveguideHeightComponent:
      return waveguide_height.get_free_dof_value(comp.second);
      break;
    case WaveguideWidthComponent:
      return waveguide_width.get_free_dof_value(comp.second);
      break;
    default:
      break;
  }
  return 0.0;
}

void WaveguideTransformation::set_free_dof(int index, double value) {
  std::pair<ResponsibleComponent, unsigned int> comp = map_free_dof_index(index);
  switch (comp.first)
  {
    case VerticalDisplacementComponent:
      vertical_shift.set_free_dof_value(comp.second, value);
      return;
      break;
    case WaveguideHeightComponent:
      waveguide_height.set_free_dof_value(comp.second, value);
      return;
      break;
    case WaveguideWidthComponent:
      waveguide_width.set_free_dof_value(comp.second, value);
      return;
      break;
    default:
      break;
  }
  std::cout << "There was an error setting a free dof value." << std::endl; 
  return;
}

void WaveguideTransformation::estimate_and_initialize() {
  vertical_shift.set_constraints(0, GlobalParams.Vertical_displacement_of_waveguide, 0,0);
  vertical_shift.initialize();
  if(!GlobalParams.keep_waveguide_height_constant) {
    waveguide_height.set_constraints(GlobalParams.Height_of_waveguide, GlobalParams.Height_of_waveguide, 0,0);
    waveguide_height.initialize();
  }
  if(!GlobalParams.keep_waveguide_width_constant) {
    waveguide_width.set_constraints(GlobalParams.Width_of_waveguide, GlobalParams.Width_of_waveguide, 0,0);
    waveguide_height.initialize();
  }
}

Vector<double> WaveguideTransformation::get_dof_values() const {
  Vector<double> ret(n_dofs());
  unsigned int total_counter = 0;
  for(unsigned int i = 0; i < vertical_shift.get_n_dofs(); i++) {
    ret[total_counter] = vertical_shift.get_dof_value(i);
    total_counter ++;
  }
  if(!GlobalParams.keep_waveguide_height_constant) {
    for(unsigned int i = 0; i < waveguide_height.get_n_dofs(); i++) {
      ret[total_counter] = waveguide_height.get_dof_value(i);
      total_counter ++;
    }
  }
  if(!GlobalParams.keep_waveguide_width_constant) {
    for(unsigned int i = 0; i < waveguide_width.get_n_dofs(); i++) {
      ret[total_counter] = waveguide_width.get_dof_value(i);
      total_counter ++;
    }
  }
  return ret;
}

unsigned int WaveguideTransformation::n_free_dofs() const {
  unsigned int ret = vertical_shift.n_free_dofs;
  if(!GlobalParams.keep_waveguide_height_constant) {
    ret += waveguide_height.n_free_dofs;
  }
  if(!GlobalParams.keep_waveguide_width_constant) {
    ret += waveguide_width.n_free_dofs;
  }
  return ret;
}

void WaveguideTransformation::Print() const {
  // TODO
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int WaveguideTransformation::n_dofs() const {
  unsigned int ret = vertical_shift.n_dofs;
  if(!GlobalParams.keep_waveguide_height_constant) {
    ret += waveguide_height.n_dofs;
  }
  if(!GlobalParams.keep_waveguide_width_constant) {
    ret += waveguide_width.n_dofs;
  }
  return ret;
}

Tensor<2, 3, double>
WaveguideTransformation::get_Space_Transformation_Tensor(Position &position) {
  Tensor<2, 3, double> J_loc = get_J(position);
  Tensor<2, 3, double> ret;
  ret[0][0] = 1;
  ret[1][1] = 1;
  ret[2][2] = 1;
  return (J_loc * ret * transpose(J_loc)) / determinant(J_loc);
}

/**
 * Position ret;
  if(GlobalParams.keep_waveguide_width_constant) {
    ret[0] = coord[0];
  } else {
    ret[0] = coord[0] / waveguide_width.evaluate_at(coord[2]);
  }
  if(GlobalParams.keep_waveguide_height_constant) {
    ret[1] = coord[1] - vertical_shift.evaluate_at(coord[2]);
  } else {
    ret[1] = (coord[1] / waveguide_height.evaluate_at(coord[2])) - vertical_shift.evaluate_at(coord[2]);
  }
  ret[2] = coord[2];
  return ret;
  */

Tensor<2,3,double> WaveguideTransformation::get_J(Position &in_p) {
  Tensor<2,3,double> ret = I;
  const double z = in_p[2];
  if(GlobalParams.keep_waveguide_height_constant && GlobalParams.keep_waveguide_width_constant) {
    // Only shift down vertically
    ret[1][2] = -vertical_shift.evaluate_derivative_at(z);
  } else {
    const double y = in_p[1];
    const double h = waveguide_height.evaluate_at(z);
    const double dh = waveguide_height.evaluate_derivative_at(z);
    const double dm = vertical_shift.evaluate_derivative_at(z);
    if(GlobalParams.keep_waveguide_width_constant) {
      // Vertical shift and vertical stretching of the waveguide (variable height)
      // f(y) = (y / waveguide_height.evaluate_at(z)) - vertical_shift.evaluate_at(z);
      ret[1][2] = - dm - y * dh / h*h;
    } else {
      // Vertical shift, vertical stretching and horizontal stretching of the waveguide (variable height and width)
      const double w = waveguide_width.evaluate_at(z);
      const double dw = waveguide_width.evaluate_derivative_at(z);
      const double x = in_p[0];
      ret[0][2] = - x  * dw / (w*w); 
      ret[1][2] = - dm - y * dh / h*h;
    }
  }

  return ret;
}

Tensor<2,3,double> WaveguideTransformation::get_J_inverse(Position &in_p) {
  Tensor<2,3,double> ret = get_J(in_p);
  return invert(ret);
}

std::pair<ResponsibleComponent, unsigned int> WaveguideTransformation::map_free_dof_index(unsigned int in_index) const {
  std::pair<ResponsibleComponent, unsigned int> ret;
  if(in_index < vertical_shift.get_n_free_dofs()) {
    ret.first = ResponsibleComponent::VerticalDisplacementComponent;
  } else {
    unsigned int local_index = in_index - vertical_shift.get_n_free_dofs();
    if(local_index < waveguide_height.get_n_free_dofs()) {
      ret.first = ResponsibleComponent::WaveguideHeightComponent;
    } else {
      ret.first = ResponsibleComponent::WaveguideWidthComponent;
    }
  }
  switch (ret.first)
  {
    case VerticalDisplacementComponent:
      ret.second = in_index;
      break;
    case WaveguideHeightComponent:
      ret.second = in_index - vertical_shift.get_n_free_dofs();
      break;
    case WaveguideWidthComponent:
      ret.second = in_index - vertical_shift.get_n_free_dofs() - waveguide_height.get_n_free_dofs();
      break;
    default:
      break;
  }
  return ret;
}

std::pair<ResponsibleComponent, unsigned int> WaveguideTransformation::map_dof_index(unsigned int in_index) const {
  std::pair<ResponsibleComponent, unsigned int> ret;
  if(in_index < vertical_shift.get_n_dofs()) {
    ret.first = ResponsibleComponent::VerticalDisplacementComponent;
  } else {
    unsigned int local_index = in_index - vertical_shift.get_n_dofs();
    if(local_index < waveguide_height.get_n_dofs()) {
      ret.first = ResponsibleComponent::WaveguideHeightComponent;
    } else {
      ret.first = ResponsibleComponent::WaveguideWidthComponent;
    }
  }

  switch (ret.first)
  {
    case VerticalDisplacementComponent:
      ret.second = in_index;
      break;
    case WaveguideHeightComponent:
      ret.second = in_index - vertical_shift.get_n_dofs();
      break;
    case WaveguideWidthComponent:
      ret.second = in_index - vertical_shift.get_n_dofs() - waveguide_height.get_n_dofs();
      break;
    default:
      break;
  }
  return ret;
}


