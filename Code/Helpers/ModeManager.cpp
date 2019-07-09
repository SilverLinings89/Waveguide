/*
 * ModeManager.cpp
 * \brief this class is supposed to handle general modal computations. It can
 * compute the analytic shape of a mode for circular connectors as well as
 * loading data from an input file for rectangular connectors. \date 23.03.2017
 * \author Pascal Kraft
 */

#ifndef CODE_HELPERS_MODEMANAGER_CPP_
#define CODE_HELPERS_MODEMANAGER_CPP_

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <deal.II/base/logstream.h>
#include "ModeManager.h"
#include "../Core/NumericProblem.h"

ModeManager::ModeManager() {
    in_prepared = false;
    out_prepared = false;
}

void ModeManager::load() {
    // in_circular = GlobalParams.M_C_Shape == ConnectorType::Circle;
    in_circular = true;

    // out_circular = GlobalParams.M_C_Shape == ConnectorType::Circle;
    out_circular = true;
    v_in = 2 * GlobalParams.Phys_V;
    v_out = 2 * GlobalParams.Phys_V;
    return;
}

void ModeManager::prepare_mode_in() {
    if (in_circular) {
        u_in = get_us(v_in);
    }
    in_prepared = true;
}

void ModeManager::prepare_mode_out() {
    if (out_circular) {
        u_out = get_us(v_out);
    }
    out_prepared = true;
}

double ModeManager::get_input_component(int in_component,
                                        dealii::Point<3, double> position,
                                        int mode) {
    if (!in_prepared) {
        prepare_mode_in();
    }
    if (in_circular) {
        double r = std::sqrt(position[0] * position[0] + position[1] * position[1]);
        if (in_component == 0) {
            if (r < u_in[mode]) {
                return J_n(r, mode) / J_n(u_in[mode], mode);
            } else {
                return K_n(r, mode) / K_n(u_in[mode], mode);
            }
        } else {
            return 0.0;
        }
    } else {
        return 0.0;
    }
}

double ModeManager::get_output_component(int in_component,
                                         dealii::Point<3, double> position,
                                         int mode) {
    if (!out_prepared) {
        prepare_mode_out();
    }
    if (out_circular) {
        double r = std::sqrt(position[0] * position[0] + position[1] * position[1]);
        if (in_component == 0) {
            if (r < u_out[mode]) {
                return J_n(r, mode) / J_n(u_out[mode], mode);
            } else {
                return K_n(r, mode) / K_n(u_out[mode], mode);
            }
        } else {
            return 0.0;
        }
    } else {
        return 0.0;
    }
}

double ModeManager::bessi0(double x) {
    const static double p1 = 1.0e0;
    const static double p2 = 3.5156229e0;
    const static double p3 = 3.0899424e0;
    const static double p4 = 1.2067492e0;
    const static double p5 = 0.2659732e0;
    const static double p6 = 0.360768e-1;
    const static double p7 = 0.45813e-2;
    const static double q1 = 0.39894228e0;
    const static double q2 = 0.1328592e-1;
    const static double q3 = 0.225319e-2;
    const static double q4 = -0.157565e-2;
    const static double q5 = 0.916281e-2;
    const static double q6 = -0.2057706e-1;
    const static double q7 = 0.2635537e-1;
    const static double q8 = -0.1647633e-1;
    const static double q9 = 0.392377e-2;
    if (abs(x) < 3.75) {
        double y = (x / 3.75) * (x / 3.75);
        return p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7)))));
    } else {
        double ax = std::abs(x);
        double y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) *
               (q1 +
                y * (q2 +
                     y * (q3 +
                          y * (q4 +
                               y * (q5 +
                                    y * (q6 + y * (q7 + y * (q8 + y * q9))))))));
    }
}

double ModeManager::bessi1(double x) {
    const static double p1 = 0.5e0;
    const static double p2 = 0.87890594e0;
    const static double p3 = 0.51498869e0;
    const static double p4 = 0.15084934e0;
    const static double p5 = 0.2658733e-1;
    const static double p6 = 0.301532e-2;
    const static double p7 = 0.32411e-3;
    const static double q1 = 0.39894228e0;
    const static double q2 = -0.3988024e-1;
    const static double q3 = -0.362018e-2;
    const static double q4 = 0.163801e-2;
    const static double q5 = -0.1031555e-1;
    const static double q6 = 0.2282967e-1;
    const static double q7 = -0.2895312e-1;
    const static double q8 = 0.1787654e-1;
    const static double q9 = -0.420059e-2;
    if (std::abs(x) < 3.75) {
        double y = (x / 3.75) * (x / 3.75);
        return x *
               (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7))))));
    } else {
        double ax = std::abs(x);
        double y = 3.75 / ax;
        double ret =
                (std::exp(ax) / std::sqrt(ax)) *
                (q1 +
                 y * (q2 +
                      y * (q3 +
                           y * (q4 +
                                y * (q5 + y * (q6 + y * (q7 + y * (q8 + y * q9))))))));
        if (x < 0) ret *= -1.0;
        return ret;
    }
}

double ModeManager::J_0(double x) {
    const static double p1 = 1.e0;
    const static double p2 = -.1098628627E-2;
    const static double p3 = .2734510407e-4;
    const static double p4 = -.2073370639e-5;
    const static double p5 = .2093887211e-6;
    const static double q1 = -.1562499995e-1;
    const static double q2 = .1430488765e-3;
    const static double q3 = -.6911147651e-5;
    const static double q4 = .7621095161e-6;
    const static double q5 = -.934945152e-7;
    const static double r1 = 57568490574.e0;
    const static double r2 = -13362590354.e0;
    const static double r3 = 651619640.7e0;
    const static double r4 = -11214424.18e0;
    const static double r5 = 77392.33017e0;
    const static double r6 = -184.9052456e0;
    const static double s1 = 57568490411.e0;
    const static double s2 = 1029532985.e0;
    const static double s3 = 9494680.718e0;
    const static double s4 = 59272.64853e0;
    const static double s5 = 267.8532712e0;
    const static double s6 = 1.e0;
    if (std::abs(x) < 8) {
        double y = x * x;
        return (r1 + y * (r2 + y * (r3 + y * (r4 + y * (r5 + y * r6))))) /
               (s1 + y * (s2 + y * (s3 + y * (s4 + y * (s5 + y * s6)))));
    } else {
        double ax = std::abs(x);
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - .785398164;
        return std::sqrt(.636619772 / ax) *
               (std::cos(xx) * (p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)))) -
                z * std::sin(xx) * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5)))));
    }
}

double ModeManager::J_1(double x) {
    const static double r1 = 72362614232.e0;
    const static double r2 = -7895059235.e0;
    const static double r3 = 242396853.1e0;
    const static double r4 = -2972611.439e0;
    const static double r5 = 15704.48260e0;
    const static double r6 = -30.16036606e0;
    const static double q1 = .04687499995e0;
    const static double q2 = -.2002690873e-3;
    const static double q3 = .8449199096e-5;
    const static double q4 = -.88228987e-6;
    const static double q5 = .105787412e-6;
    const static double p1 = 1.e0;
    const static double p2 = .183105e-2;
    const static double p3 = -.3516396496e-4;
    const static double p4 = .2457520174e-5;
    const static double p5 = -.240337019e-6;
    const static double s1 = 144725228442.e0;
    const static double s2 = 2300535178.e0;
    const static double s3 = 18583304.74e0;
    const static double s4 = 99447.43394e0;
    const static double s5 = 376.9991397e0;
    const static double s6 = 1.e0;
    if (std::abs(x) < 8) {
        double y = x * x;
        return x * (r1 + y * (r2 + y * (r3 + y * (r4 + y * (r5 + y * r6))))) /
               (s1 + y * (s2 + y * (s3 + y * (s4 + y * (s5 + y * s6)))));
    } else {
        double ax = std::abs(x);
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 2.356194491;
        return std::sqrt(.636619772 / ax) *
               (std::cos(xx) * (p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)))) -
                z * std::sin(xx) * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5))))) *
               std::copysign(1., x);
    }
}

double ModeManager::K_0(double x) {
    const static double p1 = -0.57721566e0;
    const static double p2 = 0.42278420e0;
    const static double p3 = 0.23069756e0;
    const static double p4 = 0.3488590e-1;
    const static double p5 = 0.262698e-2;
    const static double p6 = 0.10750e-3;
    const static double p7 = 0.74e-5;
    const static double q1 = 1.25331414e0;
    const static double q2 = -0.7832358e-1;
    const static double q3 = 0.2189568e-1;
    const static double q4 = -0.1062446e-1;
    const static double q5 = 0.587872e-2;
    const static double q6 = -0.251540e-2;
    const static double q7 = 0.53208e-3;
    if (x < 2.0) {
        double y = x * x / 4.0;
        return (-log(x / 2.0) * bessi0(x)) +
               (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7))))));
    } else {
        double y = 2.0 / x;
        return (exp(-x) / sqrt(x)) *
               (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * (q6 + y * q7))))));
    }
}

double ModeManager::K_1(double x) {
    const static double p1 = 1.0e0;
    const static double p2 = 0.15443144e0;
    const static double p3 = -0.67278579e0;
    const static double p4 = -0.18156897e0;
    const static double p5 = -0.1919402e-1;
    const static double p6 = -0.110404e-2;
    const static double p7 = -0.4686e-4;
    const static double q1 = 1.25331414e0;
    const static double q2 = 0.23498619e0;
    const static double q3 = -0.3655620e-1;
    const static double q4 = 0.1504268e-1;
    const static double q5 = -0.780353e-2;
    const static double q6 = 0.325614e-2;
    const static double q7 = -0.68245e-3;
    if (x < 2.0) {
        double y = x * x / 4.0;
        return (log(x / 2.0) * bessi1(x)) +
               (1.0 / x) *
               (p1 +
                y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7))))));
    } else {
        double y = 2.0 / x;
        return (exp(-x) / sqrt(x)) *
               (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * (q6 + y * q7))))));
    }
}

double ModeManager::K_n(double x, int n) {
    double tox = 2.0 / x;
    double bkp;
    double bkm = K_0(x);
    double bk = K_1(x);
    if (n == 0) {
        return K_0(x);
    }
    if (n == 1) {
        return K_1(x);
    }
    for (int j = 1; j < n; j++) {
        bkp = bkm + j * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}

double ModeManager::J_n(double x, int n) {
    double bessj = 0;
    const int IACC = 40;
    const double BIGN0 = 1.e10;
    const double BIGNI = 1.e-10;
    if (n == 0) {
        return J_0(x);
    }
    if (n == 1) {
        return J_1(x);
    }
    double ax = std::abs(x);
    if (ax == 0) {
        return 0;
    } else {
        if (ax > (float) n) {
            double tox = 2.0 / ax;
            double bjm = J_0(ax);
            double bj = J_1(ax);
            for (int j = 1; j < n; j++) {
                double bjp = j * tox * bj - bjm;
                bjm = bj;
                bj = bjp;
            }
            bessj = bj;
        } else {
            double tox = 2.0 / ax;
            int m = 2 * ((n + (int) std::sqrt((float) IACC * n)) / 2);
            double bessj = 0;
            int jsum = 0;
            double sum = 0.0;
            double bjp = 0.0;
            double bj = 1.0;
            for (int j = m; j > 1; j--) {
                double bjm = j * tox * bj - bjp;
                bjp = bj;
                bj = bjm;
                if (std::abs(bj) > BIGN0) {
                    bj = bj * BIGNI;
                    bjp = bjp * BIGNI;
                    bessj = bessj * BIGNI;
                    sum = sum * BIGNI;
                }
                if (jsum != 0) {
                    sum = sum + bj;
                }
                jsum = 1 - jsum;
                if (j == n) {
                    bessj = bjp;
                }
            }
            sum = 2.0 * sum - bj;
            bessj /= sum;
        }
    }
    if (x < 0 && n % 2 == 1) {
        bessj = -bessj;
    }
    return bessj;
}

double ModeManager::get_rhs(double u, double v) {
    double p = std::sqrt(v * v - u * u);
    return p * K_0(p) / K_1(p);
}

double ModeManager::get_lhs(double u) { return -u * J_0(u) / J_1(u); }

std::vector<double> ModeManager::get_us(double v) {
    std::vector<double> locations;
    bool last_smaller = false;
    double lhs, rhs;
    double u = 0.1;
    lhs = get_lhs(u);
    rhs = get_rhs(u, v);
    last_smaller = lhs < rhs;
    for (u = 0.2; u < v; u += 0.1) {
        double n_lhs = get_lhs(u);
        double n_rhs = get_rhs(u, v);
        bool new_smaller = n_lhs < n_rhs;
        if (new_smaller != last_smaller) {
            locations.push_back(u);
        }
        last_smaller = new_smaller;
        lhs = n_lhs;
        rhs = n_rhs;
    }

    std::vector<double> us;
    for (unsigned int i = 0; i < locations.size(); i++) {
        double u_left = locations[i] - 0.1;
        double u_right = locations[i];
        double l_lhs = get_lhs(u_left);
        double l_rhs = get_rhs(u_left, v);

        double j = 0;
        while (j < 10) {
            double update_lhs = get_lhs((u_left + u_right) / 2.0);
            double update_rhs = get_rhs((u_left + u_right) / 2.0, v);
            if ((update_lhs - update_rhs) * (l_lhs - l_rhs) < 0) {
                u_right = (u_left + u_right) / 2.0;
            } else {
                u_left = (u_left + u_right) / 2.0;
                l_lhs = update_lhs;
                l_rhs = update_rhs;
            }
            j++;
        }
        if (std::abs(l_lhs - l_rhs) < 1) {
            us.push_back((u_left + u_right) / 2.0);
        }
    }
    for (unsigned int i = 0; i < us.size(); i++) {
        deallog << "u_" << i + 1 << ": " << us[i] << std::endl;
    }
    return us;
}

double ModeManager::get_u0(double v) {
    std::vector<double> locations;
    bool last_smaller = false;
    double lhs, rhs;
    double u = 0.1;
    lhs = get_lhs(u);
    rhs = get_rhs(u, v);
    last_smaller = lhs < rhs;
    for (u = 0.2; u < v; u += 0.1) {
        double n_lhs = get_lhs(u);
        double n_rhs = get_rhs(u, v);
        bool new_smaller = n_lhs < n_rhs;
        if (new_smaller != last_smaller) {
            locations.push_back(u);
        }
        last_smaller = new_smaller;
        lhs = n_lhs;
        rhs = n_rhs;
    }

    for (unsigned int i = 0; i < locations.size(); i++) {
        std::cout << "sign change near " << locations[i] << std::endl;
    }

    std::vector<double> us;
    for (unsigned int i = 0; i < locations.size(); i++) {
        double u_left = locations[i] - 0.1;
        double u_right = locations[i];
        double l_lhs = get_lhs(u_left);
        double l_rhs = get_rhs(u_left, v);

        double j = 0;
        while (j < 10) {
            double update_lhs = get_lhs((u_left + u_right) / 2.0);
            double update_rhs = get_rhs((u_left + u_right) / 2.0, v);
            if ((update_lhs - update_rhs) * (l_lhs - l_rhs) < 0) {
                u_right = (u_left + u_right) / 2.0;
            } else {
                u_left = (u_left + u_right) / 2.0;
                l_lhs = update_lhs;
                l_rhs = update_rhs;
            }
            j++;
        }
        if (std::abs(l_lhs - l_rhs) < 1) {
            us.push_back((u_left + u_right) / 2.0);
        }
    }

    for (unsigned int i = 0; i < us.size(); i++) {
        std::cout << "intersection at " << us[i] << std::endl;
    }

    if (us.size() > 0) {
        return us[0];
    } else {
        return -1;
    }
}

#endif /* CODE_HELPERS_MODEMANAGER_CPP_ */
