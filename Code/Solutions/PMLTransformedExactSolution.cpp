#include "./PMLTransformedExactSolution.h"

PMLTransformedExactSolution::PMLTransformedExactSolution(BoundaryId in_main_id, double in_additional_coordinate): Function<3, ComplexNumber>(3) {
    main_boundary = in_main_id;
    additional_coordinate = in_additional_coordinate;
    non_pml_layer_thickness = GlobalParams.PML_thickness / GlobalParams.PML_N_Layers;
    base_solution = GlobalParams.source_field;

}

double PMLTransformedExactSolution::compute_scaling_factor(const Position & in_p) const {
  const unsigned int comp_of_pml = main_boundary / 2;
  // Compute the pml progress variable:
  double prog = fraction_of_pml_direction(in_p)[comp_of_pml];
  // this is not precise because there is scaling of sigma but I use constant sigma here.
  double sigma_eff = GlobalParams.PML_Sigma_Max;
  // This is the effective distance the signal has traveled in the PML.
  double distance = GlobalParams.PML_thickness - GlobalParams.PML_thickness / GlobalParams.PML_N_Layers;
  distance = distance * prog;
  if(GlobalParams.PML_skaling_order != 0) {
    sigma_eff =  std::pow(prog, GlobalParams.PML_skaling_order + 1) / GlobalParams.PML_skaling_order;
  }
  return std::exp(-1.0 * sigma_eff * distance) ;
}

ComplexNumber PMLTransformedExactSolution::value(const Position &p, const unsigned int component) const {
  const unsigned int comp_of_pml = main_boundary / 2;
  ComplexNumber base_val = base_solution->value(p,component);
  if(component != comp_of_pml) {
      return base_val;
  }
  base_val *= compute_scaling_factor(p);
  return base_val;

}

void PMLTransformedExactSolution::vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const {
  dealii::Vector<ComplexNumber> ret;
  ret.reinit(3);
  base_solution->vector_value(p, ret);
  double scaling_factor = compute_scaling_factor(p);
  for(unsigned int i= 0; i < 3; i++) {
    ret[i] *= scaling_factor;
  }
  value = ret;
}

dealii::Tensor<1, 3, ComplexNumber> PMLTransformedExactSolution::curl(const Position &) const {
  dealii::Tensor<1, 3, ComplexNumber> ret;
  /** 
  NumericVectorLocal curls = base_solution->curl(in_p);
  double scaling_factor = compute_scaling_factor(in_p);
  for(unsigned int i = 0; i < 3; i++) {
    ret[i] *= scaling_factor;
  }
  **/
  return ret;
}

dealii::Tensor<1, 3, ComplexNumber> PMLTransformedExactSolution::val(const Position &in_p) const {
  dealii::Tensor<1, 3, ComplexNumber> ret;
  NumericVectorLocal vals;
  base_solution->vector_value(in_p, vals);
  double scaling_factor = compute_scaling_factor(in_p);
  for(unsigned int i = 0; i < 3; i++) {
    ret[i] = vals[i] * scaling_factor;
  }
  return ret;
}

std::array<double, 3> PMLTransformedExactSolution::fraction_of_pml_direction(const Position & in_p) const {
  std::array<double, 3> ret;
  if(in_p[0] < Geometry.local_x_range.first) {
    ret[0] = (Geometry.local_x_range.first - in_p[0] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[0] > Geometry.local_x_range.second) {
      ret[0] = (in_p[0] - Geometry.local_x_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[0] = 0.0;
    }
  }
  if(in_p[1] < Geometry.local_y_range.first) {
    ret[1] = (Geometry.local_y_range.first - in_p[1] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[1] > Geometry.local_y_range.second) {
      ret[1] = (in_p[1] - Geometry.local_y_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[1] = 0.0;
    }
  }
  if(in_p[2] < Geometry.local_z_range.first) {
    ret[2] = (Geometry.local_z_range.first - in_p[0] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[2] > Geometry.local_z_range.second) {
      ret[2] = (in_p[2] - Geometry.local_z_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[2] = 0.0;
    }
  }
  return ret;
}