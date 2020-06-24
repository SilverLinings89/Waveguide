/*
 * PointSourceField.h
 *
 *  Created on: Jun 9, 2020
 *      Author: kraft
 */

#ifndef CODE_HELPERS_POINTSOURCEFIELD_H_
#define CODE_HELPERS_POINTSOURCEFIELD_H_

#include <deal.II/base/function.h>

class PointSourceField: public dealii::Function<3, std::complex<double>> {
public:
  double k = 1;
  PointSourceField();
  virtual ~PointSourceField();
  std::complex<double> value(const dealii::Point<3> &p,
      const unsigned int component = 0) const override;
  void vector_value(const dealii::Point<3> &p,
      dealii::Vector<std::complex<double>> &vec) const override;
  void vector_curl(const dealii::Point<3> &p,
      dealii::Vector<std::complex<double>> &vec);
};

#endif /* CODE_HELPERS_POINTSOURCEFIELD_H_ */
