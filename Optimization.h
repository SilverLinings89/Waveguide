#ifndef OptimizationFlag
#define OptimizationFlag

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include "WaveguideStructure.h"
#include "Waveguide.h"
#include "Parameters.h"

using namespace dealii;

class Optimization {
	public:
		const int dofs; // (sectors +1) *3 -6
		const Parameters System_Parameters;
		Waveguide<SparseMatrix<double>, Vector<double> > &waveguide;
		WaveguideStructure &structure;

		Optimization( Parameters , Waveguide<SparseMatrix<double>, Vector<double> >  & , WaveguideStructure &);
		void run();

};

#endif
