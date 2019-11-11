//
// Created by kraft on 08.11.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H
#define WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H


#include <deal.II/base/point.h>
#include "HSIESurface.h"

class HSIEPolynomial {
    std::vector<std::complex<double>> a, da;
    std::complex<double> k0;
    std::complex<double> evaluate( std::complex<double> x);
    std::complex<double> evaluate_dx ( std::complex<double> x);
    void update_derivative();
    static void computeDandI(unsigned int, std::complex<double>);

public:
    explicit HSIEPolynomial(DofData& in_dof, std::complex<double> k0);
    explicit HSIEPolynomial(std::vector<std::complex<double>> in_a, std::complex<double> k0);

    static bool matricesLoaded;
    static dealii::FullMatrix<std::complex<double>> D;
    static dealii::FullMatrix<std::complex<double>> I;

    HSIEPolynomial applyD();
    HSIEPolynomial applyI();
};


#endif //WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H
