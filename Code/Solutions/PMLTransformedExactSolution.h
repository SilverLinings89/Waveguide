#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "./ExactSolution.h"

class PMLTransformedExactSolution: public dealii::Function<3, ComplexNumber> {
 private:
  dealii::Function<3, ComplexNumber> * base_solution;
  BoundaryId main_boundary;
  double additional_coordinate;
  double non_pml_layer_thickness;

 public:
  PMLTransformedExactSolution( BoundaryId in_main_id, double in_additional_coordinate);
  
  std::vector<std::string> split(std::string) const;

  ComplexNumber value(const Position &p, const unsigned int component) const;

  void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;

  dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;

  dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
  
  std::array<double, 3> fraction_of_pml_direction(const Position & in_p) const;

  double compute_scaling_factor(const Position & in_p) const;
};
