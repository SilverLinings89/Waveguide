#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "Waveguide.h"
#include "staticfunctions.cpp"
#include "WaveguideStructure.h"

using namespace dealii;

template<typename MatrixType, typename VectorType >
Waveguide<MatrixType, VectorType>::Waveguide (Parameters &param, WaveguideStructure &in_structure)
  :
  fe (FE_Nedelec<3> (0), 2),
  dof_handler (triangulation),
  prm(param),
  log_data(),
  log_constraints(std::string("constraints.log"), log_data),
  log_assemble(std::string("assemble.log"), log_data),
  log_precondition(std::string("precondition.log"), log_data),
  log_total(std::string("total.log"), log_data),
  log_solver(std::string("solver.log"), log_data),
  structure(in_structure)
{
	assembly_progress = 0;
	int i = 0;
	bool dir_exists = true;
	while(dir_exists) {
		std::stringstream out;
		out << "solutions/run";
		out << i;
		solutionpath = out.str();
		struct stat myStat;
		const char *myDir = solutionpath.c_str();
		if ((stat(myDir, &myStat) == 0) && (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
			i++;
		} else {
			dir_exists = false;
		}
	}

	mkdir(solutionpath.c_str(), ACCESSPERMS);
	std::cout << "Will write solutions to " << solutionpath << std::endl;
	is_stored = false;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_out () {
	double ret = 0.0;
	double z = -prm.PRM_M_R_ZLength/2;
	for (int  i = 0; i < StepsR; i++) {
		double r = ((structure.r_0 + structure.r_1)/2.0) / (StepsR + 1) * (i+1);
		for(int j = 0; j < StepsPhi; j++) {
			double phi = 2 * PI * j / StepsPhi;
			Point<3, double> position(r * cos(phi), r * sin(phi), z);
			Vector<double> result(6);
			VectorTools::point_value(dof_handler, solution, position, result);
			double Q1 = structure.getQ1(position);
			double Q2 = structure.getQ2(position);
			ret += (result[0] / Q1) * ( TEMode00( position , 0) / Q1);
			ret += (result[1] / Q2) * ( TEMode00( position , 1) / Q2);
			ret += (result[3] / Q1) * ( TEMode00( position , 0) / Q1);
			ret += (result[4] / Q2) * ( TEMode00( position , 1) / Q2);
		}
	}
	return ret;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_in () {
	double ret = 0.0;
		double z = -prm.PRM_M_R_ZLength/2;
		for (int  i = 0; i < StepsR; i++) {
			double r = ((structure.r_0 + structure.r_1)/2.0) / (StepsR + 1) * (i+1);
			for(int j = 0; j < StepsPhi; j++) {
				double phi = 2 * PI * j / StepsPhi;
				Point<3, double> position(r * cos(phi), r * sin(phi), z);
				Vector<double> result(6);
				VectorTools::point_value(dof_handler, solution, position, result);
				double Q1 = structure.getQ1(position);
				double Q2 = structure.getQ2(position);
				ret += (result[0] / Q1)*(TEMode00(position,0) / Q1);
				ret += (result[1] / Q2)*(TEMode00(position,1) / Q2);
				ret += (result[3] / Q1)*(TEMode00(position,0) / Q1);
				ret += (result[4] / Q2)*(TEMode00(position,1) / Q2);
			}
		}
	return ret;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_overall () {
	double quality_in	= evaluate_in();
	double quality_out	= evaluate_out();

	return quality_out/quality_in;
}


template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::store() {
	storage.reinit(dof_handler.n_dofs());
	for(unsigned int i = 0; i < dof_handler.n_dofs(); i++){
		storage[i] = solution[i];
	}
	if(!is_stored) is_stored = true;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {
	std::cout << "get_Tensor_1" << std::endl;
	Tensor<2,3, std::complex<double>> ret;
	for(int i = 0; i<3; i++ ){
		for(int j = 0; j<3; j++) {
			ret[i][j] = 0.0;
		}
	}
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);

	double omegaepsilon0 = (2* PI / prm.PRM_M_W_Lambda) * c ;
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = prm.PRM_M_R_XLength * 1.0 * prm.PRM_M_BC_Mantle/100.0;
		sigmax = pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_KappaXMax);
		sx.imag( sigmax / (prm.PRM_M_W_EpsilonOut * omegaepsilon0));
		S1 /= sx;
		S2 *= sx;
		S3 *= sx;
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = prm.PRM_M_R_YLength * 1.0 * prm.PRM_M_BC_Mantle/100.0;
		sigmay = pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_KappaYMax);
		sy.imag( sigmay / (prm.PRM_M_W_EpsilonOut * omegaepsilon0));
		S1 *= sy;
		S2 /= sy;
		S3 *= sy;
	}
	if(PML_in_Z(position)){
		double r,d, sigmaz;
		r = PML_Z_Distance(position);
		d = prm.PRM_M_R_ZLength * 1.0 * ((position(2)<0)? prm.PRM_M_BC_XYin : prm.PRM_M_BC_XYout)/100.0;
		sigmaz = pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(d/r , prm.PRM_M_BC_M) * prm.PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / ((System_Coordinate_in_Waveguide(position))?prm.PRM_M_W_EpsilonIn : prm.PRM_M_W_EpsilonOut) * omegaepsilon0 );
		S1 *= sz;
		S2 *= sz;
		S3 /= sz;
	}

	if(inverse) {
		std::complex<double> temp(1.0, 0.0);
		S1 = temp / S1;
		S2 = temp / S2;
		S3 = temp / S3;
	}

	ret[0][0] = S1;
	ret[1][1] = S2;
	ret[2][2] = S3;

	if(inverse) {
		if(epsilon) {
			if(System_Coordinate_in_Waveguide(position)) {
				ret /= prm.PRM_M_W_EpsilonIn;
			} else {
				ret /= prm.PRM_M_W_EpsilonOut;
			}
			ret /= Eps0;
		} else {
			ret /= Mu0;
		}
	} else {
		if(epsilon) {
			if(System_Coordinate_in_Waveguide(position) ) {
				ret *= prm.PRM_M_W_EpsilonIn;
			} else {
				ret *= prm.PRM_M_W_EpsilonOut;
			}
			ret *= Eps0;
		} else {
			ret *= Mu0;
		}
	}
	Tensor<2,3, double> transformation = structure.TransformationTensor(position[0], position[1], position[2]);
	Tensor<2,3, std::complex<double>> ret2;

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			ret[i][j] = std::complex<double>(0.0, 0.0);
			for(int k = 0; k < 3; k++) {
				ret2[i][j] += ret[i][k] * transformation[k][j];
			}
		}
	}
	std::cout << "get_Tensor_2" << std::endl;

	return ret2;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Transpose_Tensor(Tensor<2,3, std::complex<double>> input) {
	Tensor<2,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		for(int j = 0; j<3; j++){
			ret[i][j].real(input[i][j].real());
			ret[i][j].imag( - input[i][j].imag());
		}
	}
	return ret;
}

template<typename MatrixType, typename VectorType >
Tensor<1,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Transpose_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_X(Point<3> &p) {
	return  p(0) > ((prm.PRM_M_R_XLength / 2.0) - (prm.PRM_M_R_XLength * prm.PRM_M_BC_Mantle/100.0)) ||  p(0) < (-(prm.PRM_M_R_XLength / 2.0) + (prm.PRM_M_R_XLength * prm.PRM_M_BC_Mantle/100.0));
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_Y(Point<3> &p) {
	return  p(1) > ((prm.PRM_M_R_YLength / 2.0) - (prm.PRM_M_R_YLength * prm.PRM_M_BC_Mantle/100.0)) ||  p(1) < (-(prm.PRM_M_R_YLength / 2.0) + (prm.PRM_M_R_YLength * prm.PRM_M_BC_Mantle/100.0));
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_Z(Point<3> &p) {
	return  (p(2) > ((prm.PRM_M_R_ZLength / 2.0) - (prm.PRM_M_R_ZLength * prm.PRM_M_BC_XYout/100.0)) ) ||  (p(2) < (-(prm.PRM_M_R_ZLength / 2.0) + (prm.PRM_M_R_ZLength * prm.PRM_M_BC_XYin/100.0)  )  && !System_Coordinate_in_Waveguide(p));
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_X_Distance(Point<3> &p){
	if(p(0) >0){
		return p(0) - ((prm.PRM_M_R_XLength / 2.0) - (prm.PRM_M_R_XLength* prm.PRM_M_BC_Mantle/100.0));
	} else {
		return -(p(0) + ((prm.PRM_M_R_XLength / 2.0) - (prm.PRM_M_R_XLength* prm.PRM_M_BC_Mantle/100.0)));
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Y_Distance(Point<3> &p){
	if(p(1) >0){
		return p(1) - ((prm.PRM_M_R_YLength / 2.0) - (prm.PRM_M_R_YLength* prm.PRM_M_BC_Mantle/100.0));
	} else {
		return -(p(1) + ((prm.PRM_M_R_YLength / 2.0) - (prm.PRM_M_R_YLength* prm.PRM_M_BC_Mantle/100.0)));
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Z_Distance(Point<3> &p){
	if(p(2) >0){
		return p(2) - ((prm.PRM_M_R_ZLength / 2.0) - (prm.PRM_M_R_ZLength * prm.PRM_M_BC_XYout/100));
	} else {
		return -(p(2) + ((prm.PRM_M_R_ZLength / 2.0) - (prm.PRM_M_R_ZLength * prm.PRM_M_BC_XYin/100)));
	}
}


template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::make_grid ()
{
	log_total.start();
	const double outer_radius = 1.0;
	GridGenerator::subdivided_hyper_cube (triangulation, 5, -outer_radius, outer_radius);

	static const CylindricalManifold<3, 3> round_description(2, 0.0001);
	unsigned int temp = 1;
	triangulation.set_manifold (temp, round_description);
	Triangulation<3>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += Distance2D(Point<3> (cell->vertex(j)));
		if (distance_from_center < 3 ) {
			cell->set_all_manifold_ids(1);
		}
	}

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += Distance2D(Point<3> (cell->vertex(j)));
		if (distance_from_center < 1.2) {
			cell->set_manifold_id(0);
		}
	}

	GridTools::transform(& Triangulation_Stretch_X, triangulation);
	GridTools::transform(& Triangulation_Stretch_Y, triangulation);
	GridTools::transform(& Triangulation_Stretch_Z, triangulation);


	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		if(cell->at_boundary()){
			for(int j = 0; j<6; j++){
				if(cell->face(j)->at_boundary()){
					Point<3> ctr =cell->face(j)->center(true, false);
					if(System_Coordinate_in_Waveguide(ctr)){
						if(ctr(2) < 0) cell->face(j)->set_all_boundary_indicators(1);
						else cell->face(j)->set_all_boundary_indicators(2);
					}
				}
			}
		}
	}


	if(prm.PRM_D_Refinement == "global") triangulation.refine_global (prm.PRM_D_XY);

	cell = triangulation.begin_active();
	double l = (double)prm.PRM_M_R_ZLength / (prm.PRM_A_Threads*2.0);
	for (; cell!=endc; ++cell){

		int temp  = (int) (((cell->center(true, false))[2] + (prm.PRM_M_R_ZLength/2)) / l);
		if( temp >= 2* prm.PRM_A_Threads) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
		cell->set_subdomain_id(temp);
	}


	if(prm.PRM_O_Grid) {
		if(prm.PRM_O_VerboseOutput) std::cout<< "Writing Mesh data to file \"grid-3D.vtk\"" << std::endl;
		mesh_info(triangulation, "grid-3D.vtk");
		if(prm.PRM_O_VerboseOutput) std::cout<< "Done" << std::endl;

	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::setup_system ()
{
	if(prm.PRM_O_VerboseOutput && prm.PRM_O_Dofs) {
		std::cout << "Distributing Degrees of freedom." << std::endl;
	}
	dof_handler.distribute_dofs (fe);
	if(prm.PRM_O_VerboseOutput) {
		std::cout << "Renumbering DOFs (Cuthill_McKee...)" << std::endl;
	}

	DoFRenumbering::Cuthill_McKee (dof_handler);
	if(prm.PRM_O_Dofs) {
		std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(prm.PRM_O_VerboseOutput) {
		std::cout << "Calculating compressed Sparsity Pattern..." << std::endl;
	}

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);

	system_matrix.reinit( sparsity_pattern );
	solution.reinit ( dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	if(prm.PRM_O_VerboseOutput) {
			std::cout << "Done." << std::endl;
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_part ( unsigned int in_part) {
	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	FullMatrix<double>	cell_matrix_real (dofs_per_cell, dofs_per_cell);
	Vector<double>		cell_rhs (dofs_per_cell);
	cell_rhs = 0;
	Tensor<2,3, std::complex<double>> 		epsilon, mu;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();

	for (; cell!=endc; ++cell)
	{
		if(cell->subdomain_id() == in_part) {
			fe_values.reinit (cell);
			quadrature_points = fe_values.get_quadrature_points();
			cell_matrix_real = 0;

			for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
			{
				epsilon = get_Tensor(quadrature_points[q_index],  false, true);
				mu = get_Tensor(quadrature_points[q_index], true, false);
				const double JxW = fe_values.JxW(q_index);
				for (unsigned int i=0; i<dofs_per_cell; i++){
					Tensor<1,3, std::complex<double>> I_Curl;
					Tensor<1,3, std::complex<double>> I_Val;
					for(int k = 0; k<3; k++){
						I_Curl[k].imag(fe_values[imag].curl(i, q_index)[k]);
						I_Curl[k].real(fe_values[real].curl(i, q_index)[k]);
						I_Val[k].imag(fe_values[imag].value(i, q_index)[k]);
						I_Val[k].real(fe_values[real].value(i, q_index)[k]);
					}
					for (unsigned int j=0; j<dofs_per_cell; j++){
						Tensor<1,3, std::complex<double>> J_Curl;
						Tensor<1,3, std::complex<double>> J_Val;
						for(int k = 0; k<3; k++){
							J_Curl[k].imag(fe_values[imag].curl(j, q_index)[k]);
							J_Curl[k].real(fe_values[real].curl(j, q_index)[k]);
							J_Val[k].imag(fe_values[imag].value(j, q_index)[k]);
							J_Val[k].real(fe_values[real].value(j, q_index)[k]);
						}

						std::complex<double> x = (mu * I_Curl) * Transpose_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Transpose_Vector(J_Val)) *(omega * omega)*JxW ;
						if(x.real() != 0) {
							cell_matrix_real[i][j] += x.real();
						}
					}
				}
			}

			cell->get_dof_indices (local_dof_indices);

			cm.distribute_local_to_global(cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, false );
			//cm.distribute_local_to_global(cell_matrix_imag, cell_rhs, local_dof_indices,system_matrix.block(0,1), system_rhs.block(1), true );

	    }
	}
	assembly_progress ++;
	std::cout << "Progress: " << 100 * assembly_progress/(prm.PRM_A_Threads*2) << " %" << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_system ()
{
	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	if(prm.PRM_O_VerboseOutput) {
		std::cout << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		std::cout << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
	}

	log_data.Dofs = dof_handler.n_dofs();
	log_constraints.start();

	//starting to calculate Constraint Matrix for boundary values;
	//VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 0 , cm , StaticMappingQ1<3>::mapping);
	VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 1 , cm , StaticMappingQ1<3>::mapping);
	//VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 2 , cm , StaticMappingQ1<3>::mapping);

	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();

	for (; cell!=endc; ++cell)
	{
		bool At_Boundary = false;
		bool Is_At_One = false;
		for(int i = 0; i < 6; i++) {
			if(cell->face(i)->at_boundary()){
				At_Boundary = true;
				if(cell->face(i)->boundary_indicator() == 1) Is_At_One = true;
			}
		}
		if(At_Boundary && !Is_At_One) {
			std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
			cell->get_dof_indices(local_dof_indices);
			unsigned int i;
			for (i = 0; i < dofs_per_cell; ++i) {
				cm.add_line(local_dof_indices[i]);
				cm.set_inhomogeneity(local_dof_indices[i], 0);
			}
		}
	}

	cm.close();
	//cm.distribute(solution);
	log_constraints.stop();

	log_assemble.start();
	std::cout << "Starting Assemblation process" << std::endl;
	Threads::TaskGroup<void> task_group1;
	for (int i = 0; i < prm.PRM_A_Threads; ++i) {
		task_group1 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i);
	}
	task_group1.join_all ();

	Threads::TaskGroup<void> task_group2;
	for (int i = 0; i < prm.PRM_A_Threads; ++i) {
		task_group2 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i+1);
	}
	task_group2.join_all ();

	std::cout<<system_rhs.l2_norm()<<std::endl;
	log_assemble.stop();

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::estimate_solution() {
	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(! (cell->at_boundary())) {
			std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					Point<3, double> p = ((cell->face(i))->line(j))->center(false, false);
					Point<3, double> direction = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
					const std::complex<double> z(0.0, omega * (p(2)- prm.PRM_M_R_ZLength/2.0));
					double d2 = Distance2D(p);
					const std::complex<double> result = - exp(z) * exp(-d2*d2/2);
					solution[local_dof_indices[0]] = result.real();
					solution[local_dof_indices[1]] = result.imag();
				}
			}
		}
	}
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::timerupdate() {
	log_precondition.stop();
	log_solver.start();
}

template<>
void Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> >::solve () {
	SolverControl          solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, true, true);
	log_precondition.start();

	if(is_stored) {
		solution.reinit(dof_handler.n_dofs());
		for(unsigned int i = 0; i < dof_handler.n_dofs(); i++){
			solution[i] = storage[i];
		}
		if(prm.PRM_S_Solver == "GMRES") {
			SolverGMRES<Vector<double> > solver (solver_control, SolverGMRES<Vector<double> >::AdditionalData(prm.PRM_S_GMRESSteps, true));

			if(prm.PRM_S_Preconditioner == "Block_Jacobi"){
				PreconditionBlockJacobi<SparseMatrix<double>, double> block_jacobi;
				block_jacobi.initialize(system_matrix, PreconditionBlock<SparseMatrix<double>, double>::AdditionalData(log_data.Dofs / prm.PRM_S_PreconditionerBlockCount));
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, block_jacobi);
			}

			if(prm.PRM_S_Preconditioner == "Identity") {
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
			}

			if(prm.PRM_S_Preconditioner == "Jacobi"){
				PreconditionJacobi<SparseMatrix<double>> pre_jacobi;
				pre_jacobi.initialize(system_matrix, PreconditionJacobi<SparseMatrix<double>>::AdditionalData());
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, pre_jacobi);
			}

			if(prm.PRM_S_Preconditioner == "SOR"){
				PreconditionSOR<SparseMatrix<double> > plu;
				plu.initialize(system_matrix, .6);
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, plu);
			}

			if(prm.PRM_S_Preconditioner == "SSOR"){
				PreconditionSSOR<SparseMatrix<double> > plu;
				plu.initialize(system_matrix, .6);
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, plu);
			}

			if(prm.PRM_S_Preconditioner == "ILU"){
				SparseILU<double> ilu;
				ilu.initialize(system_matrix, SparseILU<double>::AdditionalData());
				timerupdate();
				solver.solve (system_matrix, solution, system_rhs, ilu);
			}
		}
	} else {
		SparseDirectUMFPACK  A_direct;
		A_direct.initialize(system_matrix);
		timerupdate();
		A_direct.vmult(solution, system_rhs);
	}


	log_solver.stop();
	cm.distribute(solution);
}

template <typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::solve ()
{

	if(typeid(VectorType) == typeid(Vector<double>)) {

	}

	if(typeid(MatrixType) == typeid(TrilinosWrappers::SparseMatrix)) {

	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::init_loggers () {
	log_data.PML_in 				= 	prm.PRM_M_BC_XYin;
	log_data.PML_out 				=	prm.PRM_M_BC_XYout;
	log_data.PML_mantle 			=	prm.PRM_M_BC_Mantle;
	log_data.ParamSteps 			=	prm.PRM_S_Steps;
	log_data.Precondition_BlockSize = 	0;
	log_data.Precondition_weight 	=	0;
	log_data.Solver_Precision 		=	prm.PRM_S_Precision;
	log_data.XLength				=	prm.PRM_M_R_XLength;
	log_data.YLength 				= 	prm.PRM_M_R_YLength;
	log_data.ZLength 				= 	prm.PRM_M_R_ZLength;
	log_data.preconditioner 		=	prm.PRM_S_Preconditioner;
	log_data.solver 				= 	prm.PRM_S_Solver;
	log_data.Dofs 					=	0;
	log_constraints.Dofs			=	true;
	log_constraints.PML_in			=	log_constraints.PML_mantle	= log_constraints.PML_out		= true;
	log_assemble.Dofs				=	true;
	log_precondition.Dofs			=	true;
	log_precondition.preconditioner = 	log_precondition.cputime	= true;
	log_solver.Dofs					=	true;
	log_solver.solver				=	log_solver.preconditioner	= log_solver.Solver_Precision	= log_solver.cputime	= true;
	log_total.Dofs					=	log_total.solver			= log_total.Solver_Precision	= log_total.cputime		= true;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::output_results () const
{


	DataOut<3> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");

	data_out.build_patches ();

	//std::ofstream output ("solution.gpl");
	std::ofstream outputvtk (solutionpath + "/solution.vtk");
	data_out.write_vtk(outputvtk);
	//data_out.write_gnuplot (output);

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::run ()
{
	init_loggers ();
	// read_values ();
	make_grid ();
	setup_system ();
	assemble_system ();
	estimate_solution();
	solve ();
	// output_results ();
	log_total.stop();
}

#endif
