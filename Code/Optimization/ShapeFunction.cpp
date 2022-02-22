
#include <iostream>
#include <cmath>
#include "./ShapeFunction.h"
#include "../Core/Types.h"

unsigned int ShapeFunction::compute_n_free_dofs(unsigned int in_n_sectors, bool in_is_lower_value_constrained, bool in_is_upper_value_constrained, bool in_is_lower_derivative_constrained, bool in_is_upper_derivative_constrained) {
    int ret = ShapeFunction::compute_n_dofs(in_n_sectors);
    ret -= 1;
    if(in_is_lower_derivative_constrained) {
        ret -= 1;
    }
    if(in_is_upper_derivative_constrained) {
        ret -= 1;
    }
    if(in_is_lower_value_constrained) {
        ret -= 1;
    }
    if(in_is_upper_value_constrained) {
        ret -= 1;
    }
    if(ret < 0) {
        std::cout << "The shape function is underdetermined. Add more sectors." << std::endl;
    }
    return std::abs(ret);
}

unsigned int ShapeFunction::compute_n_dofs(unsigned int in_n_sectors) {
    return in_n_sectors+3;
}


ShapeFunction::ShapeFunction(double in_z_min, double in_z_max, unsigned int in_n_sectors,bool in_is_lower_value_constrained, bool in_is_upper_value_constrained, bool in_is_lower_derivative_constrained, bool in_is_upper_derivative_constrained):
is_lower_derivative_constrained(in_is_lower_derivative_constrained),
is_upper_derivative_constrained(in_is_upper_derivative_constrained),
is_lower_value_constrained(in_is_lower_value_constrained),
is_upper_value_constrained(in_is_upper_value_constrained),
sector_length((in_z_max - in_z_min) / (double)in_n_sectors),
n_free_dofs(ShapeFunction::compute_n_free_dofs(in_n_sectors, in_is_lower_value_constrained, in_is_upper_value_constrained, in_is_lower_derivative_constrained, in_is_upper_derivative_constrained)),
n_dofs(ShapeFunction::compute_n_dofs(in_n_sectors))
{
    dof_values.resize(n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_values[i] = 0;
    }
    z_min = in_z_min;
    z_max = in_z_max;
}

double ShapeFunction::evaluate_at(double z) {
    if(z <= z_min) {
        return dof_values[0];
    }
    if(z >= z_max) {
        return dof_values[n_dofs-1];
    }
    double ret = dof_values[0];
    double z_temp = z_min;
    unsigned int index = 1;
    while(z_temp + sector_length < z+FLOATING_PRECISION) {
        ret += 0.5 * sector_length * (dof_values[index + 1] - dof_values[index]);
        ret += sector_length * dof_values[index];
        index++;
        z_temp += sector_length;
    }
    double delta_z = z - z_temp;
    if(std::abs(delta_z) <= FLOATING_PRECISION) {
        return ret;
    }
    ret += 0.5 * delta_z * (dof_values[index + 1] - dof_values[index]) * (delta_z/sector_length);
    ret += delta_z * dof_values[index];
    return ret;
}

double ShapeFunction::evaluate_derivative_at(double z) {
    if(z <= z_min) {
        return dof_values[1];
    }
    if(z >= z_max) {
        return dof_values[n_dofs-2];
    }
    unsigned int index = 1;
    double z_temp = z_min;
    while(z_temp + sector_length < z) {
        index ++;
        z_temp += sector_length;
    }
    double delta_z = z - z_temp;
    if(std::abs(delta_z) < FLOATING_PRECISION) {
        return dof_values[index];
    } else {
        return dof_values[index] + (dof_values[index + 1] - dof_values[index]) * ( delta_z / sector_length );
    }
}

void ShapeFunction::set_constraints(double in_f_0, double in_f_1, double in_df_0, double in_df_1) {
    if(is_lower_value_constrained) {
        f_0 = in_f_0;
    }
    if(is_lower_derivative_constrained) {
        df_0 = in_df_0;
    }
    if(is_upper_value_constrained) {
        f_1 = in_f_1;
    }
    if(is_upper_derivative_constrained) {
        df_1 = in_df_1;
    }
}

void ShapeFunction::update_constrained_values() {
    if(is_lower_value_constrained) {
        dof_values[0] = f_0;
    }
    if(is_lower_derivative_constrained) {
        dof_values[1] = df_0;
    }
    if(is_upper_derivative_constrained) {
        dof_values[n_dofs-2] = df_1;
    }
    if(is_upper_value_constrained) {
        dof_values[n_dofs-1] = f_1;
    }
    if(is_upper_value_constrained && is_lower_value_constrained && is_upper_derivative_constrained && is_lower_derivative_constrained) {
        double f_y_min2 = evaluate_at(z_max - sector_length - sector_length);
        dof_values[n_dofs-3] = (dof_values[n_dofs - 1] - f_y_min2) / sector_length - (0.5 * (dof_values[n_dofs - 4] + dof_values[n_dofs - 2]));
    }
}

void ShapeFunction::set_free_values(std::vector<double> in_dof_values) {
    if(in_dof_values.size() != n_free_dofs) {
        std::cout << "Provided wrong number of degrees of freedom." << std::endl;
    }
    unsigned int shift = 0;
    if(!is_lower_value_constrained) {
        shift += 1;
        dof_values[0] = in_dof_values[0];
    }
    if(!is_lower_derivative_constrained) {
        dof_values[1] = in_dof_values[shift];
        shift += 1;
    }

    for(unsigned int i = 2; i < dof_values.size()-3; i++) {
        dof_values[i] = in_dof_values[shift+i-2];
    }
    int down_shift = 0;
    if(!is_upper_value_constrained) {
        down_shift = 1;
        dof_values[dof_values.size() - 1] = in_dof_values[in_dof_values.size() - 1];
    }
    if(!is_upper_derivative_constrained) {
        dof_values[dof_values.size() - 2] = in_dof_values[in_dof_values.size() - 1 - down_shift];
    }
    update_constrained_values();
}
