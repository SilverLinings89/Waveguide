//
// Created by kraft on 08.11.19.
//

#include "HSIEPolynomial.h"
#include "../Helpers/staticfunctions.h"

std::complex<double> HSIEPolynomial::evaluate(std::complex<double> x_in) {
    std::complex<double> ret(a[0]);
    std::complex<double> x = x_in;
    for( unsigned long i = 1; i < this->a.size(); i++) {
        ret += a[i] * x;
        x = x * x_in;
    }
    return ret;
}

std::complex<double> HSIEPolynomial::evaluate_dx(std::complex<double> x_in) {
    std::complex<double> ret(da[0]);
    std::complex<double> x = x_in;
    for(unsigned long i = 1; i < this->da.size(); i++) {
        ret += da[i] * x;
        x = x * x_in;
    }
    return ret;
}

HSIEPolynomial::HSIEPolynomial(DofData& in_dof) {
    this->a = std::vector<std::complex<double>>();
    this->da = std::vector<std::complex<double>>();
    for(int i = 0; i < in_dof.hsie_order - 1; i++) {
        this->a.emplace_back(0.0,0.0);
    }
    if(in_dof.is_real) {
        this->a.emplace_back(1.0,0.0);
    } else {
        this->a.emplace_back(0.0,1.0);
    }
    this->update_derivative();
}

HSIEPolynomial::HSIEPolynomial(std::vector<std::complex<double>> in_a) {
    this->a = in_a;
    this->update_derivative();
}

HSIEPolynomial HSIEPolynomial::applyD() {
    std::vector<std::complex<double>> n_a;
    for(long int i = 0; i < this->a.size(); i++) {
        std::complex<double> component(0,0);
        for(long int j = 0; j < this->a.size(); j++) {
            component += this->a[j] * matrixD(i,j,this->k0);
        }
        n_a.push_back(component);
    }
    return HSIEPolynomial(n_a);
}

HSIEPolynomial HSIEPolynomial::applyI() {
    // TODO: Hier fehlt noch die Implementiereung mit I statt D.
    std::vector<std::complex<double>> n_a;
    for(long int i = 0; i < this->a.size(); i++) {
        std::complex<double> component(0,0);
        for(long int j = 0; j < this->a.size(); j++) {
            component += this->a[j] * matrixD(i,j,this->k0);
        }
        n_a.push_back(component);
    }
    return HSIEPolynomial(n_a);
}

void HSIEPolynomial::update_derivative() {
    this->da = std::vector<std::complex<double>>();
    for(long int i = 1; i < this->a.size(); i++) {
        this->da.emplace_back(i*this->a[i].real(),i*this->a[i].imag());
    }
}
