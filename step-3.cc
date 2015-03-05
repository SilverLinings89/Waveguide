#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

class ParameterReader : public Subscriptor
{
public: 
	ParameterReader	(ParameterHandler &);
	void read_parameters	(const std::string);

private: 
	void declare_parameters();
	ParameterHandler &prm;
};

ParameterReader::ParameterReader	( ParameterHandler &prmhandler) : 	prm(prmhandler) {}

void ParameterReader::declare_parameters	()
{
	prm.enter_subsection("Output");
	{
		prm.declare_entry("Output Grid", "false", Patterns::Bool() , "Determines if Grid should be written to .eps file for visualization.");
		prm.declare_entry("Output Dofs", "false", Patterns::Bool() , "Determines if details about Degrees of freedom should be written to the console.");
		prm.declare_entry("Output Active Cells", "false", Patterns::Bool() , "Determines if the number of active cells should be written to the console.");
	}
	prm.leave_subsection();

	prm.enter_subsection("Measures");
	{
		prm.enter_subsection("Connectors");
		{
			prm.declare_entry("Type in", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Type out", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Radius in", "1.5", Patterns::Double(0), "Radius / Diameter for Circle / Square input connector. Ellipse not implemented.");
			prm.declare_entry("Radius out", "1.5", Patterns::Double(0), "Radius / Diameter for Circle / Square output connector. Ellipse not implemented.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Region");
		{
			prm.declare_entry("XLength", "10", Patterns::Integer(0), "Length of the system in x-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("YLength", "10", Patterns::Integer(0), "Length of the system in y-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("ZLength", "450", Patterns::Integer(0), "Length of the system in z-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
		}
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
		{
			prm.declare_entry("Delta", "0.0", Patterns::Double(0), "Offset between the two connectors measured in micrometres.");
			prm.declare_entry("epsilon in", "2.2", Patterns::Double(0), "Material-Property of the optical fiber (optical thickness).");
			prm.declare_entry("epsilon out", "1.0", Patterns::Double(0), "Material-Property of environment of the fiber (optical thickness).");
			prm.declare_entry("Lambda", "0.6328", Patterns::Double(0), "Vacuum-wavelength of the incoming wave.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
		{
			prm.declare_entry("Type", "PML", Patterns::Selection("PML|HSIE"), "The way the output-connector is modeled. HSIE uses the Hardy-space infinite element for setting boundary conditions but isn't implemented yet.");
			prm.declare_entry("XY in" , "10.0" , Patterns::Double(0), "Thickness of the PML area on the side of the input connector.");
			prm.declare_entry("XY out" , "10.0" , Patterns::Double(0), "Thickness of the PML area on the side of the output connector.");
			prm.declare_entry("Mantle" , "4.0" , Patterns::Double(0), "Thickness of the PML area on 4 non-connector sides, the mantle.");
		}
		prm.leave_subsection();

	}
	prm.leave_subsection();

	prm.enter_subsection("Discretization");
	{
		prm.declare_entry("refinement", "global", Patterns::Selection("global|adaptive"), "This value describes if the XY-plane discretization should be refined homogeneously or adaptively. The latter is not implemented yet.");
		prm.declare_entry("XY", "4", Patterns::Integer(1), "Number of refinement steps used in the XY-plane.");
		prm.declare_entry("Z" , "100", Patterns::Integer(1), "Number of layers in the z-direction.");
	}
	prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string inputfile) {
	declare_parameters();
	prm.read_input(inputfile);
}


class Step3
{
	public:
		Step3 (ParameterHandler &);
		void run ();

	private:
		void read_values ();
		void make_grid ();
		void setup_system ();
		void assemble_system ();
		void solve ();
		void output_results () const;

		Triangulation<2>     triangulation;
		FE_Q<2>              fe;
		DoFHandler<2>        dof_handler;

		SparsityPattern      sparsity_pattern;
		SparseMatrix<double> system_matrix;
		ParameterHandler &prm;

		bool PRM_O_Grid, PRM_O_Dofs, PRM_O_ActiveCells;
		std::string PRM_M_C_TypeIn, PRM_M_C_TypeOut;
		double PRM_M_C_RadiusIn, PRM_M_C_RadiusOut;
		int PRM_M_R_XLength, PRM_M_R_YLength, PRM_M_R_ZLength;
		double PRM_M_W_Delta, PRM_M_W_EpsilonIn, PRM_M_W_EpsilonOut, PRM_M_W_Lambda;
		std::string PRM_M_BC_Type;
		double PRM_M_BC_XYin, PRM_M_BC_XYout, PRM_M_BC_Mantle;
		std::string PRM_D_Refinement;
		int PRM_D_XY, PRM_D_Z;
		Vector<double>       solution;
		Vector<double>       system_rhs;
};

Step3::Step3 (ParameterHandler &param)
  :
  fe (1),
  dof_handler (triangulation),
  prm(param)
{ }

void Step3::read_values() {
	prm.enter_subsection("Output");
	{
		PRM_O_Grid	=	prm.get_bool("Output Grid");
		PRM_O_Dofs	=	prm.get_bool("Output Dofs");
		PRM_O_ActiveCells	=	prm.get_bool("Output Active Cells");
	}
	prm.leave_subsection();

	prm.enter_subsection("Measures");
	{
		prm.enter_subsection("Connectors");
		{
			PRM_M_C_TypeIn	= prm.get("Type in");
			PRM_M_C_TypeOut	= prm.get("Type out");
			PRM_M_C_RadiusIn	= prm.get_double("Radius in");
			PRM_M_C_RadiusOut	= prm.get_double("Radius out");
		}
		prm.leave_subsection();

		prm.enter_subsection("Region");
		{
			PRM_M_R_XLength = prm.get_integer("XLength");
			PRM_M_R_YLength = prm.get_integer("YLength");
			PRM_M_R_ZLength = prm.get_integer("ZLength");
		}
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
		{
				PRM_M_W_Delta = prm.get_double("Delta");
				PRM_M_W_EpsilonIn = prm.get_double("epsilon in");
				PRM_M_W_EpsilonOut = prm.get_double("epsilon out");
				PRM_M_W_Lambda = prm.get_double("Lambda");
		}
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
		{
			PRM_M_BC_Type = prm.get("Type");
			PRM_M_BC_XYin = prm.get_double("XY in");
			PRM_M_BC_XYout = prm.get_double("XY out");
			PRM_M_BC_Mantle = prm.get_double("Mantle");
		}
		prm.leave_subsection();

	}
	prm.leave_subsection();

	prm.enter_subsection("Discretization");
	{
		PRM_D_Refinement = prm.get("refinement");
		PRM_D_XY = prm.get_integer("XY");
		PRM_D_Z = prm.get_integer("Z");
	}
	prm.leave_subsection();

}

void Step3::make_grid ()
{

	const Point<2> center (0,0.0001);
	const double outer_radius = 1.0;
	GridGenerator::subdivided_hyper_cube (triangulation, 5, -outer_radius, outer_radius);
	static const SphericalManifold<2> round_description(center);
	triangulation.set_manifold (1, round_description);
	Triangulation<2>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += center.distance (cell->vertex(j));
			//std::cout << "Distance appeared: " << distance_from_center << std::endl;
		if (distance_from_center < 3 ) {
			cell->set_all_manifold_ids(1);
		}
	}

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += center.distance (cell->vertex(j));
		//std::cout << "Distance appeared: " << distance_from_center << std::endl;
		if (distance_from_center < 1.2) {
			cell->set_manifold_id(0);
		}
	}

	triangulation.refine_global (3);

}

void Step3::setup_system ()
{
	dof_handler.distribute_dofs (fe);
	std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (dof_handler.n_dofs());
	system_rhs.reinit (dof_handler.n_dofs());
}

void Step3::assemble_system ()
{

	QGauss<2>  quadrature_formula(2);

	FEValues<2> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
    {

		fe_values.reinit (cell);

		cell_matrix = 0;
		cell_rhs = 0;

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
			for (unsigned int i=0; i<dofs_per_cell; ++i)
				for (unsigned int j=0; j<dofs_per_cell; ++j)
					cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));


			for (unsigned int i=0; i<dofs_per_cell; ++i)
				cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            1 *
                            fe_values.JxW (q_index));
        }

		cell->get_dof_indices (local_dof_indices);

		for (unsigned int i=0; i<dofs_per_cell; ++i)
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

		for (unsigned int i=0; i<dofs_per_cell; ++i)
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }

	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values (dof_handler, 	0, 	ZeroFunction<2>(), 	boundary_values);
	MatrixTools::apply_boundary_values (boundary_values,
	system_matrix,
	solution,
	system_rhs);
}


void Step3::solve ()
{
	SolverControl           solver_control (1000, 1e-12);
	SolverCG<>              solver (solver_control);
	solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

}


void Step3::output_results () const
{

	DataOut<2> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");

	data_out.build_patches ();

	std::ofstream output ("solution.gpl");
	data_out.write_gnuplot (output);
}

void Step3::run ()
{
	read_values ();
	make_grid ();
	setup_system ();
	assemble_system ();
	solve ();
	output_results ();
}

int main ()
{
	ParameterHandler prm;
	ParameterReader param(prm);
	param.read_parameters("parameters.prh");
	Step3 laplace_problem(prm);
	laplace_problem.run ();
	return 0;
}

