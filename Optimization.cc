#include "WaveguideStructure.cc"


class Optimization {
	public: 
		Optimization( Parameters , Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> >  & , WaveguideStructure &);
		void run();
		Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> > waveguide;
		WaveguideStructure structure;
		const Parameters System_Parameters;
		const int dofs; // (sectors +1) *3 -6
};

Optimization::Optimization( Parameters in_System_Parameters ,Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> >  &in_wg, WaveguideStructure &in_structure)
:
		dofs((in_System_Parameters.PRM_M_W_Sectors +1)*3 -6),
		System_Parameters(in_System_Parameters)
	{
	waveguide = in_wg;
	structure = in_structure;
	
}

void Optimization::run() {
	double gradient [dofs];
	bool abort = false;
	int steps_counter = 0;
	structure.estimate_and_initialize();
	waveguide.run();
	steps_counter ++;
	waveguide.store();
	//double quality = waveguide.evaluate();
	while (steps_counter < System_Parameters.PRM_Op_MaxCases)
	for(int i = 0; i < dofs; i++) {
		//structure.set_gradient(i);
		//waveguide.load();
		waveguide.run();
		//double temp_quality = waveguide.evaluate();
		//gradient[i] = quality - temp_quality;
	}
	
	
}
