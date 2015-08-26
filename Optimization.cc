class Optimization {
	private: 
		Waveguide waveguide;
		WaveguideStructure structure;
		const Parameters System_Parameters;
		const int dofs; // (sectors +1) *3 -6 
	public: 
		Optimization( Parameters in_System_Parameters ,&Waveguide in_wg, &WaveguideStructure in_structure);
		run();
}

Optimization::Optimization( Parameters in_System_Parameters ,&Waveguide in_wg, &WaveguideStructure in_structure):System_Parameters(in_System_Paramters) , dofs((in_System_Parameters.PRM_M_W_Sectors +1 )*3 -6){
	waveguide = in_wg;
	structure = in_structure;
	
}

Optimization::run() {
	double[] gradient = new double[dofs];
	bool abort = false;
	structure.estimate();
	waveguide.run();
	waveguide.store();
	double quality = waveguide.evaluate();
	while (steps_counter < Parameters.)
	for(int i = 0; i < dofs; i++) {
		structure.set_gradient(i);
		waveguide.load();
		waveguide.run();
		double temp_quality = waveguide.evaluate();
		gradient[i] = quality - temp_quality;
	}
	
	
}