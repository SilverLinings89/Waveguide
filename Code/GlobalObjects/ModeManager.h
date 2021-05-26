#pragma once

#include <deal.II/base/point.h>

class ModeManager {
 public:
  ModeManager();

  void prepare_mode_in();

  void prepare_mode_out();

  int number_modes_in();

  int number_modes_out();

  double get_input_component(int, Position, int);

  double get_output_component(int, Position, int);

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
