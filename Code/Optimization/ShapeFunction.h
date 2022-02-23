#pragma once

#include <vector>

class ShapeFunction {

    const double sector_length = 0;
    unsigned int n_sectors;
    std::vector<double> dof_values;
    double f_0 = 0;
    double f_1 = 0;
    double df_0 = 0;
    double df_1 = 0;
    double z_min = 0;
    double z_max = 0;
    public:
    const unsigned int n_free_dofs;
    const unsigned int n_dofs;
    static unsigned int compute_n_dofs(unsigned int in_n_sectors);
    static unsigned int compute_n_free_dofs(unsigned int in_n_sectors);
    ShapeFunction(double in_z_min, double in_z_max, unsigned int in_n_sectors );
    double evaluate_at(double z) const;
    double evaluate_derivative_at(double z) const;
    void set_constraints(double in_f_0, double in_f_1, double in_df_0, double in_df_1);
    void update_constrained_values();
    void set_free_values(std::vector<double> in_dof_values);
    unsigned int get_n_dofs() const;
    unsigned int get_n_free_dofs() const;
    double get_dof_value(unsigned int index) const;
    double get_free_dof_value(unsigned int index) const;
    void initialize();
    void set_free_dof_value(unsigned int index, double value);
};