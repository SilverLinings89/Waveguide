#ifndef OPTIMIZATION_STEEPEST_DESCENT_CPP
#define OPTIMIZATION_STEEPEST_DESCENT_CPP

#include "OptimizationSteepestDescent.h"

OptimizationSteepestDescent::OptimizationSteepestDescent() {}

OptimizationSteepestDescent::~OptimizationSteepestDescent() {}

bool OptimizationSteepestDescent::perform_big_step_next(int) {
    int full_steps = residuals.size();
    int small_steps = states.size();
    return (full_steps <= small_steps);
}

double OptimizationSteepestDescent::get_small_step_step_width(int) {
    return GlobalParams.StepWidth;
}

bool OptimizationSteepestDescent::perform_small_step_next(int) {
    return !perform_big_step_next(0);
}

std::vector<double> OptimizationSteepestDescent::get_big_step_configuration() {
    std::vector<double> ret;
    if (residuals.size() == 0 && states.size() == 0) {
        return ret;
    }
    int idx = states.size() - 1;
    ret.resize(states[0].size());
    for (unsigned int i = 0; i < states[0].size(); i++) {
        ret[i] = -0.0001 * states[idx][i];
    }
    return ret;
}

#endif
