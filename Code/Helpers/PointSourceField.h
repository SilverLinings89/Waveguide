#pragma once 

#include <deal.II/base/function.h>
#include "../Core/Types.h"

class PointSourceFieldHertz: public dealii::Function<3, ComplexNumber> {
public:
  double k = 1;
  const ComplexNumber ik;
  double cell_diameter = 0.01;
  PointSourceFieldHertz(double in_k = 1.0);
  void set_cell_diameter(double diameter);
  virtual ~PointSourceFieldHertz();
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  void vector_curl(const Position &p, NumericVectorLocal &vec);
};

class PointSourceFieldCosCos: public dealii::Function<3, ComplexNumber> {
public:
  PointSourceFieldCosCos();
  virtual ~PointSourceFieldCosCos();
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  void vector_curl(const Position &p, NumericVectorLocal &vec);
};

