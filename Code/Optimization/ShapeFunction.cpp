
#include <iostream>
#include <cmath>
#include "./ShapeFunction.h"
#include "../Core/Types.h"
#include "../GlobalObjects/GlobalObjects.h"

unsigned int ShapeFunction::compute_n_free_dofs(unsigned int in_n_sectors) {
    int ret = ShapeFunction::compute_n_dofs(in_n_sectors);
    ret -= 5;
    if(ret < 0) {
        std::cout << "The shape function is underdetermined. Add more sectors." << std::endl;
    }
    return std::abs(ret);
}

unsigned int ShapeFunction::compute_n_dofs(unsigned int in_n_sectors) {
    return in_n_sectors+3;
}


ShapeFunction::ShapeFunction(double in_z_min, double in_z_max, unsigned int in_n_sectors, bool in_bad_init):
sector_length((in_z_max - in_z_min) / (2*(double)in_n_sectors)),
n_free_dofs(ShapeFunction::compute_n_free_dofs(in_n_sectors)),
n_dofs(ShapeFunction::compute_n_dofs(in_n_sectors))
{
    dof_values.resize(n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_values[i] = 0;
    }
    z_min = in_z_min;
    z_max = in_z_max/2.0;
    bad_init = in_bad_init;
}

double ShapeFunction::evaluate_at(double z) const {
    if(z <= z_min) {
        return dof_values[0];
    }
    if(z > z_max) {
        return evaluate_at(2*z_max - z);
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

double ShapeFunction::evaluate_derivative_at(double z) const {
    if(z <= z_min) {
        return dof_values[1];
    }
    if(z > z_max) {
        return - evaluate_derivative_at(2*z_max - z);
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
    f_0 = in_f_0;
    df_0 = in_df_0;
    f_1 = in_f_1;
    df_1 = in_df_1;
    update_constrained_values();
}

void ShapeFunction::update_constrained_values() {
    dof_values[0] = f_0;
    dof_values[1] = df_0;
    dof_values[n_dofs-2] = df_1;
    dof_values[n_dofs-1] = f_1;
    double f_y_min2 = evaluate_at(z_max - sector_length - sector_length);
    dof_values[n_dofs-3] = (dof_values[n_dofs - 1] - f_y_min2) / sector_length - (0.5 * (dof_values[n_dofs - 4] + dof_values[n_dofs - 2]));
}

void ShapeFunction::set_free_values(std::vector<double> in_dof_values) {
    if(in_dof_values.size() != n_free_dofs) {
        std::cout << "Provided wrong number of degrees of freedom." << std::endl;
    }
    for(unsigned int i = 0; i < in_dof_values.size(); i++) {
        dof_values[2 + i] = in_dof_values[i];
    }
    update_constrained_values();
}

void ShapeFunction::initialize() {
    std::srand(time(NULL));
    std::vector<double> initial_values;
    initial_values.resize(n_free_dofs);
    if(bad_init) {
        if(GlobalParams.Vertical_displacement_of_waveguide > FLOATING_PRECISION) {
            for(unsigned int i = 0; i < n_free_dofs; i++) {
                initial_values[i] =0;
            }
        } else {
            for(unsigned int i = 0; i < n_free_dofs; i++) {
                double local_contribution = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) - 0.5;
                initial_values[i] = dealii::Utilities::MPI::sum(local_contribution, MPI_COMM_WORLD) / GlobalParams.NumberProcesses;
            }
        }
    } else {
        for(unsigned int i = 0; i < n_free_dofs; i++) {
            initial_values[i] = (dof_values[dof_values.size()-1] - dof_values[0])/(z_max - z_min);
        }
    }
    set_free_values(initial_values);
}

unsigned int ShapeFunction::get_n_dofs() const {
    return n_dofs;
}
unsigned int ShapeFunction::get_n_free_dofs() const {
    return n_free_dofs;
}
double ShapeFunction::get_dof_value(unsigned int index) const {
    return dof_values[index];
}
double ShapeFunction::get_free_dof_value(unsigned int index) const {
    return dof_values[index + 2];
}

void ShapeFunction::set_free_dof_value(unsigned int index, double value) {
    if(index < n_free_dofs) {
        dof_values[index + 2] = value;
        update_constrained_values();
    } else {
        std::cout << "You tried to write to a constrained dof of a shape function." << std::endl;
    }
}

void ShapeFunction::print() {
    for(double x = z_min; x <= 2* z_max; x += 0.1 ) {
        std::cout << x << "\t" << evaluate_at(x)<< std::endl;
    }
}