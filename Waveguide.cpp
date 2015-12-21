#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "Waveguide.h"
#include "staticfunctions.cpp"
#include "WaveguideStructure.h"
#include "SolutionWeight.h"
#include "ExactSolution.h"
#include <sstream>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/vector_tools.templates.h>



#include <deal.II/base/std_cxx11/bind.h>

using namespace dealii;

template<typename MatrixType, typename VectorType >
Waveguide<MatrixType, VectorType>::Waveguide (Parameters &param, WaveguideStructure &in_structure)
  :
  fe (FE_Nedelec<3> (0), 2),
  dof_handler (triangulation),
  dof_handler_real(triangulation_real),
  prm(param),
  deallog(),
  log_data(),
  log_constraints(std::string("constraints.log"), log_data),
  log_assemble(std::string("assemble.log"), log_data),
  log_precondition(std::string("precondition.log"), log_data),
  log_total(std::string("total.log"), log_data),
  log_solver(std::string("solver.log"), log_data),
  structure(in_structure),
  run_number(0),
  condition_file_counter(0),
  eigenvalue_file_counter(0),
  Sectors(prm.PRM_M_W_Sectors),
  temporary_pattern_preped(false),
  real(0),
  imag(3)
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
	Dofs_Below_Subdomain[prm.PRM_M_W_Sectors];
	mkdir(solutionpath.c_str(), ACCESSPERMS);
	deallog << "Will write solutions to " << solutionpath << std::endl;
	is_stored = false;

}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_out () {
	double ret = 0.0;
	double z = prm.PRM_M_R_ZLength/2.0;
	for (int  i = 0; i < StepsR; i++) {
		double r = ((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0) / (StepsR + 1) * (i+1);
		for(int j = 0; j < i; j++) {
			double phi = 2 * GlobalParams.PRM_C_PI * j / i;
			Point<3, double> position(r * cos(phi), r * sin(phi), z);
			Vector<double> result(6);
			VectorTools::point_value(dof_handler, solution, position, result);
			double Q1 = structure.getQ1(position[2] );
			double Q2 = structure.getQ2(position[2] );
			double l1 =  (result[0] / Q1) * ( TEMode00( position , 0) / Q1);
			double l2 = (result[1] / Q2) * ( TEMode00( position , 0) / Q2);
			double l3 = (result[3] / Q1) * ( TEMode00( position , 0) / Q1);
			double l4 = (result[4] / Q2) * ( TEMode00( position , 0) / Q2);
			ret += l1*l1 + l2*l2 + l3*l3 + l4*l4;
		}
	}
	return sqrt(ret);
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_in () {
	double ret = 0.0;
	double z = -prm.PRM_M_R_ZLength/2.0 ;
	for (int  i = 0; i < StepsR; i++) {
		double r = ((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0) / (StepsR + 1) * (i+1);
		for(int j = 0; j < i; j++) {
			double phi = 2 * GlobalParams.PRM_C_PI * j / i;
			Point<3, double> position(r * cos(phi), r * sin(phi), z);
			Vector<double> result(6);
			VectorTools::point_value(dof_handler, solution, position, result);
			double Q1 = structure.getQ1(position[2] );
			double Q2 = structure.getQ2(position[2] );
			double l1 =  (result[0] / Q1) * ( TEMode00( position , 0) / Q1);
			double l2 = (result[1] / Q2) * ( TEMode00( position , 0) / Q2);
			double l3 = (result[3] / Q1) * ( TEMode00( position , 0) / Q1);
			double l4 = (result[4] / Q2) * ( TEMode00( position , 0) / Q2);
			ret += l1*l1 + l2*l2 + l3*l3 + l4*l4;
		}
	}
	return sqrt(ret);
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_overall () {
	double quality_in	= evaluate_in();
	double quality_out	= evaluate_out();
	if(!is_stored) {
		deallog << "Quality in: "<< quality_in << std::endl;
		deallog << "Quality out: "<< quality_out << std::endl;
	}
	/**
	differences.reinit(triangulation.n_active_cells());
	QGauss<3>  quadrature_formula(4);
	VectorTools::integrate_difference(dof_handler, solution, ExactSolution<3>(), differences, quadrature_formula, VectorTools::L2_norm, new SolutionWeight<3>(), 2.0);
	double L2error = differences.l2_norm();
	**/
	// if(!is_stored) deallog << "L2 Norm of the error: " << L2error << std::endl;
	result_file << "Number of Dofs: " << dof_handler.n_dofs() << std::endl;
	//result_file << "L2 Norm of the error:" << L2error << std::endl;
	result_file << "Z-Length: " << GlobalParams.PRM_M_R_ZLength << std::endl;
	result_file << "Quality in: "<< quality_in << std::endl;
	result_file << "Quality out: "<< quality_out << std::endl;
	result_file << 	100* quality_out/quality_in << "%" << std::endl;
	result_file << "Delta:" << GlobalParams.PRM_M_W_Delta << std::endl;
	result_file << "Solver: " << GlobalParams.PRM_S_Solver << std::endl;
	result_file << "Preconditioner: " << GlobalParams.PRM_S_Preconditioner << std::endl;
	result_file << "Minimal Stretch: " << structure.lowest << std::endl;
	result_file << "Maximal Stretch: " << structure.highest << std::endl;
	result_file.close();

	return quality_out/quality_in;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::store() {
	reinit_storage();
	// storage.reinit(dof_handler.n_dofs());
	for(unsigned int i = 0; i < dof_handler.n_dofs(); i++){
		storage[i] = solution[i];
	}
	if(!is_stored) is_stored = true;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {
	Tensor<2,3, std::complex<double>> ret;
	for(int i = 0; i<3; i++ ){
		for(int j = 0; j<3; j++) {
			ret[i][j] = 0.0;
		}
	}
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);

	double omegaepsilon0 = GlobalParams.PRM_C_omega * ((System_Coordinate_in_Waveguide(position))?GlobalParams.PRM_M_W_EpsilonIn : GlobalParams.PRM_M_W_EpsilonOut);
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = GlobalParams.PRM_M_R_XLength * 1.0 * GlobalParams.PRM_M_BC_Mantle/100.0;
		sigmax = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaXMax);
		sx.imag( sigmax / ( omegaepsilon0));
		S1 /= sx;
		S2 *= sx;
		S3 *= sx;
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = GlobalParams.PRM_M_R_YLength * 1.0 * GlobalParams.PRM_M_BC_Mantle/100.0;
		sigmay = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaYMax);
		sy.imag( sigmay / ( omegaepsilon0));
		S1 *= sy;
		S2 /= sy;
		S3 *= sy;
	}
	if(PML_in_Z(position)){
		double r,d, sigmaz;
		r = PML_Z_Distance(position);
		d = (position(2)<0)? GlobalParams.PRM_M_BC_XYin : GlobalParams.PRM_M_BC_XYout;
		sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / omegaepsilon0 );
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
				ret /= GlobalParams.PRM_M_W_EpsilonIn;
			} else {
				ret /= GlobalParams.PRM_M_W_EpsilonOut;
			}
			ret /= GlobalParams.PRM_C_Eps0;
		} else {
			ret /= GlobalParams.PRM_C_Mu0;
		}
	} else {
		if(epsilon) {
			if(System_Coordinate_in_Waveguide(position) ) {
				ret *= GlobalParams.PRM_M_W_EpsilonIn;
			} else {
				ret *= GlobalParams.PRM_M_W_EpsilonOut;
			}
			ret *= GlobalParams.PRM_C_Eps0;
		} else {
			ret *= GlobalParams.PRM_C_Mu0;
		}
	}
	Tensor<2,3, double> transformation = structure.TransformationTensor(position[0], position[1], position[2]);
	Tensor<2,3, std::complex<double>> ret2;

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			ret2[i][j] = std::complex<double>(0.0, 0.0);
			for(int k = 0; k < 3; k++) {
				ret2[i][j] += ret[i][k] * transformation[k][j];
			}
		}
	}
	//deallog << "get_Tensor_2" << std::endl;

	return ret2;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Conjugate_Tensor(Tensor<2,3, std::complex<double>> input) {
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
Tensor<1,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Conjugate_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

template<typename MatrixType, typename VectorType  >
bool Waveguide<MatrixType, VectorType>::PML_in_X(Point<3> &p) {
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	return p(0) < -(pmlboundary) ||p(0) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool Waveguide<MatrixType, VectorType>::PML_in_Y(Point<3> &p) {
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	return p(1) < -(pmlboundary) ||p(1) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool Waveguide<MatrixType, VectorType>::PML_in_Z(Point<3> &p) {
	return (p(2) > (GlobalParams.PRM_M_R_ZLength / 2.0 )) || ((p(2) <  -GlobalParams.PRM_M_R_ZLength / 2.0) && !(System_Coordinate_in_Waveguide(p)));
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_X_Distance(Point<3> &p){
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	if(p(0) >0){
		return p(0) - (pmlboundary) ;
	} else {
		return -p(0) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Y_Distance(Point<3> &p){
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	if(p(1) >0){
		return p(1) - (pmlboundary);
	} else {
		return -p(1) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Z_Distance(Point<3> &p){
	if(p(2) >0){
		return p(2) - (GlobalParams.PRM_M_R_ZLength / 2.0) ;
	} else {
		return -p(2) - (GlobalParams.PRM_M_R_ZLength / 2.0);
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

	if(prm.PRM_D_Refinement == "global"){
		triangulation.refine_global (prm.PRM_D_XY);
	} else {

		// Globales-Refinement hier setzen.
		triangulation.refine_global (1);
		double MaxDistFromBoundary = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusIn)*1.4/2.0;

		// semi-globales Refinement hier setzen.
		for(int i = 0; i < 1; i++) {
			cell = triangulation.begin_active();
			for (; cell!=endc; ++cell){
				if(std::abs(Distance2D(cell->center(true, false)) - (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusIn)/2.0 ) < MaxDistFromBoundary) {
					cell->set_refine_flag();
				}
			}
			triangulation.execute_coarsening_and_refinement();
			MaxDistFromBoundary *= 0.7 ;
		}

		// Refinement im Wellenleiter hier setzen.

		for(int i = 0; i < 1; i++) {
			cell = triangulation.begin_active();
			for (; cell!=endc; ++cell){
				if( Distance2D(cell->center(true, false))< (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusIn)/2.0)  {
					cell->set_refine_flag();
				}
			}
			triangulation.execute_coarsening_and_refinement();
			MaxDistFromBoundary *= 0.7 ;
		}

	}

	triangulation_real.copy_triangulation(triangulation);
	GridTools::transform(& Triangulation_Stretch_Real, triangulation_real);


	int counter = 0;
	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		if(cell->at_boundary()){
			for(int j = 0; j<6; j++){
				if(cell->face(j)->at_boundary()){
					Point<3> ctr =cell->face(j)->center(true, false);
					if(System_Coordinate_in_Waveguide(ctr)){
						if(ctr(2) < 0) {

							cell->face(j)->set_all_boundary_ids(11);
							counter ++;
						}

						else {
							cell->face(j)->set_all_boundary_ids(2);
						}
					}
				}
			}
		}
	}
	deallog<<counter << " Zellen mit Dirichlet-Werten." << std::endl;



	double l = (double)(GlobalParams.PRM_M_R_ZLength + GlobalParams.PRM_M_BC_XYin + GlobalParams.PRM_M_BC_XYout) / (Sectors);
	deallog << "Länge eines Blocks:" << l << std::endl;

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){

		int temp  = (int) (((cell->center(true, false))[2] + (GlobalParams.PRM_M_R_ZLength/(2.0)) + GlobalParams.PRM_M_BC_XYin) / l);
		if( temp >=  Sectors || temp < 0) deallog << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
		cell->set_subdomain_id(temp);
	}


	if(GlobalParams.PRM_O_Grid) {
		if(prm.PRM_O_VerboseOutput) deallog<< "Writing Mesh data to file \"grid-3D.vtk\"" << std::endl;
		mesh_info(triangulation, solutionpath + "/grid.vtk");
		if(prm.PRM_O_VerboseOutput) deallog<< "Done" << std::endl;

	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::Do_Refined_Reordering() {
	const int NumberOfDofs = dof_handler.n_dofs();
	std::vector<types::global_dof_index> dof_indices (fe.dofs_per_face);
	std::vector<int> dof_subdomain(NumberOfDofs);
	std::vector<bool> dof_at_boundary(NumberOfDofs);
	std::vector<int> CellsPerSubdomain(Sectors);
	std::vector<int> InternalBoundaryDofs(Sectors);

	for(unsigned int i = 0; i < Sectors; i++) {
		CellsPerSubdomain.push_back(0);
		InternalBoundaryDofs.push_back(0);
	}

	for(unsigned int i = 0; i < NumberOfDofs; i++) {
		dof_subdomain[i] = -1;
		dof_at_boundary[i] = false;

	}
	deallog << "Stage 1" << std::endl;
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		CellsPerSubdomain[cell->subdomain_id()] += 1;
		for(int i = 0; i < 6; i++) {
			cell->face(i)->get_dof_indices(dof_indices);
			int subdom = (cell->face(i)->center(false, false)[2] - structure.z_min ) * Sectors / (structure.z_max - structure.z_min);
			if (subdom == Sectors) subdom -= 1;
			for(int j = 0; j < fe.dofs_per_face; j++) {
				if(dof_subdomain[dof_indices[j]] != subdom && dof_subdomain[dof_indices[j]] != -1 && subdom != -1 ) {
					dof_at_boundary[dof_indices[j]] = true;
					InternalBoundaryDofs[(dof_subdomain[dof_indices[j]] > subdom) ? dof_subdomain[dof_indices[j]] : subdom] += 1;
				}
				if(dof_subdomain[dof_indices[j]] < subdom) {
					dof_subdomain[dof_indices[j]] = subdom;
				}
			}
		}
	}
	deallog << "Stage 2" << std::endl;
	Block_Sizes.reserve(Sectors);
	for(int i = 0; i < Sectors; i++ ) {
		Block_Sizes[i] = 0;
	}
	deallog << "Stage 3" << std::endl;
	for(int i = 0; i < NumberOfDofs; i++) {
		if(dof_subdomain[i] >= 0 && dof_subdomain[i] <= Sectors){
			Block_Sizes[dof_subdomain[i]] ++;
		} else {
			deallog << "A critical error was detected while reordering dofs. Falty implementation." << std::endl;
		}
	}
	deallog << "Stage 4" << std::endl;

	Dofs_Below_Subdomain.reserve(Sectors);
	Dofs_Below_Subdomain[0] = 0;
	for(int i = 1; i  < Sectors; i++) {
		Dofs_Below_Subdomain[i] = Dofs_Below_Subdomain[i-1] + Block_Sizes[i-1];
	}
	deallog << "Stage 5" << std::endl;
	unsigned int dofcountertest = 0;
	for (int i =0; i < Sectors; i++) {
		dofcountertest += Block_Sizes[i];
	}
	deallog << "Stage 6" << std::endl;
	if(dofcountertest != NumberOfDofs) {
		deallog << "There are " << dof_handler.n_dofs() << " but " << dofcountertest << " were summed up!"<<std::endl;
	}
	deallog << "Stage 7" << std::endl;

	std::vector<types::global_dof_index> New_Dofs;
	for(int i =0; i < NumberOfDofs; i++) {
		New_Dofs.push_back(i);
	}
	deallog << "Stage 8" << std::endl;
	for(int i = 0; i < NumberOfDofs; i++) {
		if(i < Dofs_Below_Subdomain[dof_subdomain[i]]) {
			bool foundmatch = false;
			for(int j = Dofs_Below_Subdomain[dof_subdomain[i]]; j < NumberOfDofs && !foundmatch; j++) {
				if(dof_subdomain[j] < dof_subdomain[i] && New_Dofs[j] == j) {
					New_Dofs[i] = j;
					New_Dofs[j] = i;
					foundmatch = true;
				}
			}
			if(!foundmatch) deallog << "Found no match for dof " << i << " which should at least be at position " << Dofs_Below_Subdomain[dof_subdomain[i]] <<"." <<std::endl;
		}
	}
	deallog << "Stage 9" << std::endl;
	for(int i = 0; i < NumberOfDofs; i++) {
		if(New_Dofs[i] < Dofs_Below_Subdomain[dof_subdomain[i]]) {
			deallog << "The ordering is incomplete!" << std::endl;
		}
	}

	deallog << "Executing dof-rednumbering now..." << std::endl;
	dof_handler.renumber_dofs(New_Dofs);
	dof_handler_real.renumber_dofs(New_Dofs);

	deallog << "Block indices: " << std::endl;

	for(int i = 1 ; i < Sectors; i++) {
		deallog <<"Dofs below Block "<< i+1<<":"<<  Dofs_Below_Subdomain[i] << std::endl;
	}
	deallog << "1" <<std::endl;

	deallog << "2" <<std::endl;
	for(int i = 0; i < Sectors; i++) {
		IndexSet temp (dof_handler.n_dofs());
		temp.clear();
		deallog << "Adding Block "<< i +1 << " from " << Dofs_Below_Subdomain[i] << " to " << Dofs_Below_Subdomain[i]+Block_Sizes[i] -1<<std::endl;
		temp.add_range(Dofs_Below_Subdomain[i],Dofs_Below_Subdomain[i]+Block_Sizes[i] -1);
		deallog << i+3 <<std::endl;
		set.push_back(temp);
	}
	deallog << "Storing details in WaveguideStructure.case_sectors..." <<std::endl;
	for(int i=0; i  < Sectors; i++) {
		structure.case_sectors[i].setLowestDof( Dofs_Below_Subdomain[i] );
		structure.case_sectors[i].setNActiveCells( CellsPerSubdomain[i] );
		structure.case_sectors[i].setNDofs( Block_Sizes[i] );
		structure.case_sectors[i].setNInternalBoundaryDofs(InternalBoundaryDofs[i]);
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::setup_system ()
{

	if(prm.PRM_O_VerboseOutput && prm.PRM_O_Dofs) {
		deallog << "Distributing Degrees of freedom." << std::endl;
	}
	dof_handler.distribute_dofs (fe);
	dof_handler_real.distribute_dofs (fe);

	if(prm.PRM_O_VerboseOutput) {
		deallog << "Renumbering DOFs (Downstream...)" << std::endl;
	}

	const Point<3> direction(0,0,1);
	DoFRenumbering::downstream(dof_handler, direction, false);
	DoFRenumbering::downstream(dof_handler_real, direction, false);
	if(prm.PRM_O_Dofs) {
		deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(prm.PRM_O_VerboseOutput) {
			deallog << "Renumbering DOFs (Custom...)" << std::endl;
	}

	Do_Refined_Reordering();


	deallog << "Reordering done." << std::endl;

	log_data.Dofs = dof_handler.n_dofs();
	log_constraints.start();

	MakeBoundaryConditions();
	DoFTools::make_hanging_node_constraints(dof_handler, cm);

	deallog << "Constructing Sparsity Pattern." << std::endl;

	cm.close();
	log_constraints.stop();
	IndexSet tempset(dof_handler.n_dofs());
	tempset.clear();
	sparsity_pattern.reinit(Sectors, Sectors);
	for ( int i = 0 ; i < Sectors; i++) {
		for ( int j = 0 ; j < Sectors; j++) {
			if(i == j)	{
				sparsity_pattern.block(i,j).reinit(Block_Sizes[i], Block_Sizes[j] );
			} else {
				if( std::abs(i - j) == 1 ) {
					sparsity_pattern.block(i,j).reinit(Block_Sizes[i], Block_Sizes[j]);
				} else {
					sparsity_pattern.block(i,j).reinit(Block_Sizes[i], Block_Sizes[j]);

				}
			}
		}
	}

	sparsity_pattern.collect_sizes();


	//DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern, cm, false);

	sparsity_pattern.compress();

	deallog << "Sparsity Pattern Construction done." << std::endl;

	reinit_all();

	cm.distribute(solution);

	if(prm.PRM_O_VerboseOutput) {
		if(!is_stored) 	deallog << "Done." << std::endl;
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_all () {
	reinit_rhs();
	reinit_solution();
	reinit_systemmatrix();
	reinit_pmlmatrix();
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_pmlmatrix () {
	pml_matrix.reinit(Sectors, Sectors);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_rhs () {
	system_rhs.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) system_rhs.block(i).reinit(Block_Sizes[i]);
	system_rhs.collect_sizes();
	// This is equivalent to system_rhs.reinit(dof_handler.n_dofs()) for a pure vector. It does the same operation on a block-system.
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_systemmatrix() {
	if(!temporary_pattern_preped) {
		temporary_pattern.copy_from(sparsity_pattern);
		temporary_pattern_preped = true;
	}
	system_matrix.reinit( temporary_pattern);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_solution() {
	solution.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) solution.block(i).reinit(Block_Sizes[i]);
	solution.collect_sizes();
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_storage() {
	storage.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) storage.block(i).reinit(Block_Sizes[i]);
	storage.collect_sizes();
}

template <>
void Waveguide<PETScWrappers::MPI::BlockSparseMatrix, PETScWrappers::MPI::BlockVector>::reinit_systemmatrix() {
	system_matrix.reinit(set, sparsity_pattern,  MPI_COMM_WORLD);
}

template <>
void Waveguide<PETScWrappers::MPI::BlockSparseMatrix, PETScWrappers::MPI::BlockVector>::reinit_rhs () {
	system_rhs.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) system_rhs.block(i).reinit(MPI_COMM_WORLD, Block_Sizes[i], Block_Sizes[i]);
	system_rhs.collect_sizes();
}

template <>
void Waveguide<PETScWrappers::MPI::BlockSparseMatrix, PETScWrappers::MPI::BlockVector>::reinit_solution() {
	solution.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) solution.block(i).reinit(MPI_COMM_WORLD, Block_Sizes[i], Block_Sizes[i]);
	solution.collect_sizes();
	//solution.reinit(MPI_COMM_WORLD, dof_handler.n_dofs(), dof_handler.n_dofs());
}

template<>
void Waveguide<PETScWrappers::MPI::BlockSparseMatrix, PETScWrappers::MPI::BlockVector>::reinit_storage() {
	storage.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) storage.block(i).reinit(MPI_COMM_WORLD, Block_Sizes[i], Block_Sizes[i]);
	storage.collect_sizes();
	//storage.reinit(MPI_COMM_WORLD, dof_handler.n_dofs(), dof_handler.n_dofs());
}

template<>
void Waveguide<TrilinosWrappers::BlockSparseMatrix, TrilinosWrappers::MPI::BlockVector>::reinit_storage() {
	storage.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) storage.block(i).reinit(set[i], MPI_COMM_WORLD);
	storage.collect_sizes();
	// storage.reinit( complete_index_set(dof_handler.n_dofs()) );
}

template<>
void Waveguide<TrilinosWrappers::BlockSparseMatrix, TrilinosWrappers::MPI::BlockVector>::reinit_rhs() {
	system_rhs.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) system_rhs.block(i).reinit(set[i], MPI_COMM_WORLD);
	system_rhs.collect_sizes();
	// system_rhs.reinit( complete_index_set(dof_handler.n_dofs()) );
}

template<>
void Waveguide<TrilinosWrappers::BlockSparseMatrix, TrilinosWrappers::MPI::BlockVector>::reinit_solution() {
	solution.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) solution.block(i).reinit(set[i], MPI_COMM_WORLD);
	solution.collect_sizes();
	//solution.reinit( complete_index_set(dof_handler.n_dofs()) );
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_part ( unsigned int in_part) {
	QGauss<3>  			 quadrature_formula(2);
	FEValues<3> 		fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	FullMatrix<double>	cell_matrix_real (dofs_per_cell, dofs_per_cell);
	Vector<double>		cell_rhs (dofs_per_cell);
	cell_rhs = 0;
	Tensor<2,3, std::complex<double>> 		epsilon, mu;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<3>::active_cell_iterator cell, endc;
	int test = 0;
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
						test ++;

						std::complex<double> x = (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.PRM_C_omega*GlobalParams.PRM_C_omega;

						cell_matrix_real[i][j] += x.real();

					}
				}
			}
			cell->get_dof_indices (local_dof_indices);

			cm.distribute_local_to_global(cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, false);

	    }
	}
	assembly_progress ++;

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_system ()
{
	reinit_rhs();
	// system_rhs.reinit(dof_handler.n_dofs());

	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	if(prm.PRM_O_VerboseOutput) {
		if(!is_stored) {
		deallog << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		deallog << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
		}
	}

	log_assemble.start();
	if(!is_stored) deallog << "Starting Assemblation process" << std::endl;
	deallog << "Sectors: " << Sectors << " Half: " << Sectors / 2 << "." << std::endl;

	Threads::TaskGroup<void> task_group1;
	int upperbound = ((Sectors % 2) == 0)? Sectors/2 : Sectors/2 + 1;
	deallog << "Upper bound" << upperbound << std::endl;
	for (int i = 0; i < upperbound; ++i) {
		task_group1 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i);
	}
	task_group1.join_all ();
	deallog << "Test" << std::endl;
	Threads::TaskGroup<void> task_group2;
	for (int i = 0; i < Sectors/2; ++i) {
		task_group2 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i+1);
	}
	task_group2.join_all ();

	if(!is_stored)  deallog<<"Assembling done. L2-Norm of RHS: "<< system_rhs.l2_norm()<<std::endl;
	log_assemble.stop();

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::MakeBoundaryConditions (){
	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
			Point<3, double> center =(cell->face(i))->center(true, false);
			if( center[0] < 0) center[0] *= (-1.0);
			if( center[1] < 0) center[1] *= (-1.0);

			if ( std::abs( center[0] - (15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7 ) ) < 0.01 ){
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					cm.add_line(local_dof_indices[0]);
					cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
					cm.add_line(local_dof_indices[1]);
					cm.set_inhomogeneity(local_dof_indices[1], 0.0);
				}
			}
			if ( std::abs( center[1] - (15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7 ) ) < 0.01 ){
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					cm.add_line(local_dof_indices[0]);
					cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
					cm.add_line(local_dof_indices[1]);
					cm.set_inhomogeneity(local_dof_indices[1], 0.0);
				}
			}
			if( std::abs(center[2] + GlobalParams.PRM_M_R_ZLength/2.0  + GlobalParams.PRM_M_BC_XYin) < 0.01 ){
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					if((cell->face(i))->line(j)->at_boundary()) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
						Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
						Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
						Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);

						double result = TEMode00(p,0);
						if(PML_in_X(p) || PML_in_Y(p)) result = 0.0;
						cm.add_line(local_dof_indices[0]);
						cm.set_inhomogeneity(local_dof_indices[0], direction[0] * result);
						cm.add_line(local_dof_indices[1]);
						cm.set_inhomogeneity(local_dof_indices[1], 0.0);

					}
				}
			}
			if( std::abs(center[2] - GlobalParams.PRM_M_R_ZLength/2.0  - GlobalParams.PRM_M_BC_XYout) < 0.01 ){
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					cm.add_line(local_dof_indices[0]);
					cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
					cm.add_line(local_dof_indices[1]);
					cm.set_inhomogeneity(local_dof_indices[1], 0.0);
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
void Waveguide<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double> >::solve () {
	SolverControl          solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, true, true);
	log_precondition.start();
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(prm.PRM_S_Solver == "MINRES") {
		condition_file.open((solutionpath + "/condition_in_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << condition_file_counter) )->str() + ".dat").c_str());
		eigenvalue_file.open((solutionpath + "/eigenvalues_in_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << eigenvalue_file_counter) )->str() + ".dat").c_str());
		SolverMinRes<BlockVector<double> > solver (solver_control, SolverMinRes<BlockVector<double> >::AdditionalData());

		if(prm.PRM_S_Preconditioner == "Identity") {
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
		}

		if(prm.PRM_S_Preconditioner == "Sweeping"){
			PreconditionSweeping<BlockSparseMatrix<double>, BlockVector<double> >::AdditionalData data( fe,  structure, dof_handler.begin_active(), dof_handler.end(), dof_handler.max_couplings_between_dofs());
			PreconditionSweeping<BlockSparseMatrix<double>, BlockVector<double> > sweep( data);
			sweep.initialize(system_matrix);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, sweep);
		}

	}

	if(prm.PRM_S_Solver == "GMRES") {
		condition_file.open((solutionpath + "/condition_in_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << condition_file_counter) )->str() + ".dat").c_str());
		eigenvalue_file.open((solutionpath + "/eigenvalues_in_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << eigenvalue_file_counter) )->str() + ".dat").c_str());
		SolverGMRES<BlockVector<double> > solver (solver_control, SolverGMRES<BlockVector<double> >::AdditionalData(prm.PRM_S_GMRESSteps, true));
		solver.connect_condition_number_slot(std_cxx11::bind(&Waveguide<dealii::BlockSparseMatrix<double> , dealii::BlockVector<double> >::print_condition, this, std_cxx11::_1), true);
		solver.connect_eigenvalues_slot(std_cxx11::bind(&Waveguide<dealii::BlockSparseMatrix<double> , dealii::BlockVector<double> >::print_eigenvalues, this, std_cxx11::_1), true);

		if(prm.PRM_S_Preconditioner == "Identity") {
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
		}
		if(prm.PRM_S_Preconditioner == "Sweeping"){
			PreconditionSweeping<BlockSparseMatrix<double>, BlockVector<double> >::AdditionalData data( fe, structure,dof_handler.begin_active(), dof_handler.end(), dof_handler.max_couplings_between_dofs());
			PreconditionSweeping<BlockSparseMatrix<double>, BlockVector<double> > sweep(data);
			sweep.initialize(system_matrix);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, sweep);
		}
	}

	if(prm.PRM_S_Solver == "UMFPACK") {
		SparseDirectUMFPACK  A_direct;
		A_direct.initialize(system_matrix);
		timerupdate();
		A_direct.vmult(solution, system_rhs);
	}

	log_solver.stop();
	cm.distribute(solution);
}

template<>
void Waveguide<dealii::TrilinosWrappers::SparseMatrix, dealii::TrilinosWrappers::MPI::Vector >::solve () {
	SolverControl          solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, true, true);
	log_precondition.start();
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(prm.PRM_S_Solver == "GMRES") {
		dealii::TrilinosWrappers::SolverGMRES solver(solver_control, dealii::TrilinosWrappers::SolverGMRES::AdditionalData(true, prm.PRM_S_GMRESSteps) );
		timerupdate();
		solver.solve(system_matrix, solution, system_rhs, dealii::TrilinosWrappers::PreconditionIdentity());
		log_solver.stop();
	}
	if(prm.PRM_S_Solver == "MINRES") {
		deallog << "The Trilinos Library does currently not offer MinRes. GMRES used instead."<<std::endl;
		dealii::TrilinosWrappers::SolverGMRES solver(solver_control,dealii::TrilinosWrappers::SolverGMRES::AdditionalData(true, prm.PRM_S_GMRESSteps) );
		timerupdate();
		solver.solve(system_matrix, solution, system_rhs, dealii::TrilinosWrappers::PreconditionIdentity());
		log_solver.stop();
	}
	if(prm.PRM_S_Solver == "UMFPACK") {
		dealii::TrilinosWrappers::SolverDirect solver(solver_control,dealii::TrilinosWrappers::SolverDirect::AdditionalData(true, "Amesos_Umfpack") );
		timerupdate();

		solver.solve(system_matrix, solution, system_rhs);
		log_solver.stop();
	}
	cm.distribute(solution);
}

template< >
void Waveguide<PETScWrappers::MPI::SparseMatrix, PETScWrappers::MPI::Vector >::solve () {
	SolverControl          solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, true, true);
	log_precondition.start();
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(prm.PRM_S_Solver == "GMRES") {
		dealii::PETScWrappers::SolverGMRES solver(solver_control,MPI_Comm(MPI_COMM_WORLD), dealii::PETScWrappers::SolverGMRES::AdditionalData(true, prm.PRM_S_GMRESSteps) );
		timerupdate();
		solver.solve(system_matrix, solution, system_rhs, dealii::PETScWrappers::PreconditionNone());
		log_solver.stop();
	} else {
		dealii::PETScWrappers::SolverCR solver(solver_control, MPI_Comm(MPI_COMM_WORLD), dealii::PETScWrappers::SolverCR::AdditionalData());
		timerupdate();
		solver.solve(system_matrix, solution, system_rhs, dealii::PETScWrappers::PreconditionNone());
		log_solver.stop();
	}
	cm.distribute(solution);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::solve ()
{


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
void Waveguide<MatrixType, VectorType>::output_results ()
{

	// evaluate_overall();

	DataOut<3> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");
	// data_out.add_data_vector(differences, "L2error");

	data_out.build_patches ();

	std::ofstream outputvtk (solutionpath + "/solution-run" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() +".vtk");
	data_out.write_vtk(outputvtk);

	DataOut<3> data_out_real;

	data_out_real.attach_dof_handler(dof_handler_real);
	data_out_real.add_data_vector (solution, "solution");
	// data_out.add_data_vector(differences, "L2error");

	data_out_real.build_patches ();

	std::ofstream outputvtk2 (solutionpath + "/solution-real" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() +".vtk");
	data_out_real.write_vtk(outputvtk2);

	std::ofstream pattern (solutionpath + "/pattern.gnu");
	sparsity_pattern.print_gnuplot(pattern);

	std::ofstream patternscript (solutionpath + "/displaypattern.gnu");
	patternscript << "set style line 1000 lw 1 lc \"black\"" <<std::endl;
	for(int i = 0; i < prm.PRM_M_W_Sectors; i++) {
		patternscript << "set arrow " << 1000 + 2*i << " from 0,-" << Dofs_Below_Subdomain[i] << " to "<<dof_handler.n_dofs()<<",-"<<Dofs_Below_Subdomain[i]<<" nohead ls 1000 front"<<std::endl;
		patternscript << "set arrow " << 1001 + 2*i  << " from " << Dofs_Below_Subdomain[i] << ",0 to " << Dofs_Below_Subdomain[i] << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
	}
	patternscript << "set arrow " << 1000 + 2*prm.PRM_M_W_Sectors << " from 0,-" << dof_handler.n_dofs() << " to "<<dof_handler.n_dofs()<<",-"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
	patternscript << "set arrow " << 1001 + 2*prm.PRM_M_W_Sectors << " from " << dof_handler.n_dofs() << ",0 to " << dof_handler.n_dofs() << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;

	patternscript << "plot \"pattern.gnu\" with dots" <<std::endl;
	patternscript.flush();

	std::ifstream source("Paramters.xml", std::ios::binary);
	std::ofstream dest(solutionpath +"/Parameters.xml", std::ios::binary);

	dest << source.rdbuf();

	source.close();
	dest.close();

}

template<typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::run ()
{
	init_loggers ();
	make_grid ();
	setup_system ();
	assemble_system ();
	solve ();
	output_results ();
	log_total.stop();
	run_number++;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::print_eigenvalues(const std::vector<std::complex<double>> &input) {
	for (unsigned int i = 0; i < input.size(); i++){
		eigenvalue_file << input.at(i).real() << "\t" << input.at(i).imag() << std::endl;
	}
	eigenvalue_file << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::print_condition(double condition) {
	condition_file << condition << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reset_changes ()
{
	reinit_all();

	cm.distribute(solution);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::rerun ()
{
	reset_changes();
	assemble_system ();
	solve ();
	output_results ();
	log_total.stop();

	run_number++;
}

#endif
