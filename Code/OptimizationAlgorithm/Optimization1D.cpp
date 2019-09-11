#ifndef OPTIMIZATION_CPP
#define OPTIMIZATION_CPP

#include "Optimization1D.h"

Optimization1D::Optimization1D() {
    steps_widths = new double[11];
    double start = 0.0001;
    // if(STEPS_PER_DOFS == 1) {
    //  steps_widths[0] = start;
    //} else {
    for (int i = 0; i < STEPS_PER_DOFS; i++) {
        steps_widths[i] = start * pow(2, i);
    }
    // for(int i = 0; i < STEPS_PER_DOFS/2; i++){
    //     steps_widths[STEPS_PER_DOFS/2 + i] = -steps_widths[i];
    // }
    //}
}

Optimization1D::~Optimization1D() { delete steps_widths; }

bool Optimization1D::perform_small_step_next(int small_steps_before) {
    if (residuals.size() == 0 && states.size() == 0) {
        return false;
    }

    return small_steps_before < STEPS_PER_DOFS;
}

double Optimization1D::get_small_step_step_width(int small_steps_before) {
    if (small_steps_before < STEPS_PER_DOFS && small_steps_before >= 0) {
        return steps_widths[small_steps_before];
    } else {
        std::cout << "Warning in Optimization1D::get_small_step_step_width(int)"
                  << std::endl;
        return -1.0;
    }
}

bool Optimization1D::perform_big_step_next(int small_steps_before) {
    if (residuals.size() == 0 && states.size() == 0) {
        return true;
    }

    return (small_steps_before >= STEPS_PER_DOFS);
}

std::vector<double> Optimization1D::get_big_step_configuration() {
    std::vector<double> ret;
    if (residuals.size() == 0 && states.size() == 0) {
        return ret;
    }
    int small_step_count = states.size();
    int big_step_count = residuals.size();

    if (big_step_count == 0 ||
        (small_step_count != STEPS_PER_DOFS * big_step_count)) {
        std::cout << "Warning in Optimization1D::get_big_step_configuration()"
                  << std::endl;
    } else {
        std::complex<double> residual = residuals[big_step_count - 1];
        double state_red = std::abs(residual);
        unsigned int ndofs = states[small_step_count - 1].size();
        ret.resize(ndofs);
        for (unsigned int i = 0; i < ndofs; i++) {
            double max = 0;
            int index = -1;
            for (unsigned int j = 0; j < STEPS_PER_DOFS; j++) {
                double magn = std::abs(
                        residual + states[small_step_count - STEPS_PER_DOFS + j][i]);
                if (magn > max && magn > state_red) {
                    max = magn;
                    index = j;
                }
            }
            if (index != -1) {
                ret[i] = steps_widths[index];
            } else {
                ret[i] = 0.0;
            }
        }
    }
    return ret;
}

#endif
