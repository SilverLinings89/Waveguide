/*
 * MainProblem.h
 *  Objects of this class are intended to solve a global Problem via MPI. They
 * only assemble parts of a problem and solve it in a distributed manner.
 *  Generally speaking this is the actual problem we are dealing with, the full
 * Maxwell System on the complete domain for example.
 * \date Jun 24, 2019
 * \author Pascal Kraft
 */

#ifndef CODE_CORE_MAINPROBLEM_H_
#define CODE_CORE_MAINPROBLEM_H_

#include "NumericProblem.h"

class MainProblem : public NumericProblem {
 public:
  MainProblem();
  virtual ~MainProblem();
};

#endif /* CODE_CORE_MAINPROBLEM_H_ */
