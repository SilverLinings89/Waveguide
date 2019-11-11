//
// Created by kraft on 08.11.19.
//

#include "HSIEPolynomial.h"
#include "../Helpers/staticfunctions.h"

bool HSIEPolynomial::matricesLoaded = false;
dealii::FullMatrix<std::complex<double>> HSIEPolynomial::D;
dealii::FullMatrix<std::complex<double>> HSIEPolynomial::I;

void HSIEPolynomial::computeDandI(unsigned int dimension, std::complex<double> k0) {
    HSIEPolynomial::D.reinit(dimension, dimension);
    for(unsigned int i = 0; i < dimension; i++) {
        for (unsigned int j = 0; j < dimension; j++) {
            HSIEPolynomial::D.set(i,j,matrixD(i, j, k0));
        }
    }
    HSIEPolynomial::I.copy_from(HSIEPolynomial::D);
    HSIEPolynomial::I.invert(HSIEPolynomial::D);
    HSIEPolynomial::matricesLoaded = true;
}

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

HSIEPolynomial::HSIEPolynomial(DofData& in_dof, std::complex<double> in_k0) {
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
    this->k0 = in_k0;
}

HSIEPolynomial::HSIEPolynomial(std::vector<std::complex<double>> in_a, std::complex<double> in_k0) {
    this->a = in_a;
    this->update_derivative();
    this->k0 = in_k0;
}

HSIEPolynomial HSIEPolynomial::applyD() {
    if(! this->matricesLoaded) {
        this->computeDandI(this->a.size(), this->k0);
    }
    std::vector<std::complex<double>> n_a;
    for(long int i = 0; i < this->a.size(); i++) {
        std::complex<double> component(0,0);
        for(long int j = 0; j < this->a.size(); j++) {
            component += this->a[j] * this->D(i,j);
        }
        n_a.push_back(component);
    }
    return HSIEPolynomial(n_a, k0);
}

HSIEPolynomial HSIEPolynomial::applyI() {
    if(! this->matricesLoaded) {
        this->computeDandI(this->a.size(), this->k0);
    }
    std::vector<std::complex<double>> n_a;
    for(long int i = 0; i < this->a.size(); i++) {
        std::complex<double> component(0,0);
        for(long int j = 0; j < this->a.size(); j++) {
            component += this->a[j] * this->I(i,j);
        }
        n_a.push_back(component);
    }
    return HSIEPolynomial(n_a, k0);
}

void HSIEPolynomial::update_derivative() {
    this->da = std::vector<std::complex<double>>();
    for(long int i = 1; i < this->a.size(); i++) {
        this->da.emplace_back(i*this->a[i].real(),i*this->a[i].imag());
    }
}
