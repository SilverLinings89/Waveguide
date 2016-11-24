#ifndef SPACETRANSFORMATION_H_
#define SPACETRANSFORMATION_H_

using namespace dealii;

class SpaceTransformation {

  const unsigned int dofs_per_layer;

  SpaceTransformation();

  Point<3> math_to_phys(Point<3> coord);

  Point<3> phys_to_math(Point<3> coord);

  bool is_identity(Point<3> coord);

  Tensor<2,3, std::complex<double>> get_epsilon(Point<3> coordinate);

  Tensor<2,3, std::complex<double>> get_mu(Point<3> coordinate);

  Tensor<2,3, double> get_transformation_tensor(Point<3> coordinate);

};

#endif SPACETRANSFORMATION_H_
 
