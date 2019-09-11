//
// Created by pascal on 11.09.19.
//

#ifndef WAVEGUIDEPROBLEM_LAGUERREFUNCTION_H
#define WAVEGUIDEPROBLEM_LAGUERREFUNCTION_H


class LaguerreFunction {
private:
    static double factorial_internal(double n);

public:
    static double evaluate(unsigned int n, unsigned int m, double x);
    static double factorial(unsigned int n);
    static unsigned int binomial_coefficient(unsigned int n, unsigned int k);
};


#endif //WAVEGUIDEPROBLEM_LAGUERREFUNCTION_H
