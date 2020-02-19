/*
 * ModeManager.h
 *
 *  \date 23.03.2017
 *  \author Pascal Kraft
 */

#ifndef CODE_HELPERS_MODEMANAGER_H_
#define CODE_HELPERS_MODEMANAGER_H_

#include <deal.II/base/point.h>

class ModeManager {
 public:
  ModeManager();

  void prepare_mode_in();

  void prepare_mode_out();

  int number_modes_in();

  int number_modes_out();

  double get_input_component(int, dealii::Point<3, double>, int);

  double get_output_component(int, dealii::Point<3, double>, int);

  void load();

 private:
  bool in_circular;
  bool out_circular;
  std::vector<double> u_in;
  double v_in;
  std::vector<double> u_out;
  double v_out;
  bool in_prepared;
  bool out_prepared;

  std::vector<double> get_us(double);

  double get_u0(double);

  double get_lhs(double);

  double get_rhs(double, double);

  double J_0(double);

  double J_1(double);

  double J_n(double, int);

  double K_0(double);

  double K_1(double);

  double K_n(double, int);

  double bessi0(double);

  double bessi1(double);
};

#endif /* CODE_HELPERS_MODEMANAGER_H_ */
