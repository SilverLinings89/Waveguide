#ifndef StaticFunctionsFlag
#define StaticFunctionsFlag

#include "ParameterReader.h"

using namespace dealii;

std::string constraints_filename 	= "constraints.log";
std::string assemble_filename 		= "assemble.log";
std::string precondition_filename 	= "precondition.log";
std::string solver_filename 		= "solver.log";
std::string total_filename 			= "total.log";

int 	StepsR 			= 10;
int 	StepsPhi 		= 10;

static int alert_counter = 0;
static std::string input_file_name = "";

void alert() {
  MPI_Barrier(MPI_COMM_WORLD);
  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout<< "Alert: " << alert_counter << std::endl;
  }
  alert_counter++;
}

static void PrepareStreams()  {
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    deallog.depth_console(10);
  } else {
    deallog.depth_console(0);
  }
  int i = 0;
  bool dir_exists = true;
  while(dir_exists) {
    std::stringstream out;
    out << "Solutions/run";
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
  i = Utilities::MPI::max(i, MPI_COMM_WORLD);
  std::stringstream out;
  out << "Solutions/run";

  out << i;
  solutionpath = out.str();
  mkdir(solutionpath.c_str(), ACCESSPERMS);

  // Copy Parameter file to the output directory in processor 0. This should be replaced with an output generator eventually.
  /**
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::ifstream source("Parameters/Parameters.xml", std::ios::binary);
    std::ofstream dest(solutionpath +"/Parameters.xml", std::ios::binary);
    dest << source.rdbuf();
    source.close();
    dest.close();
  }
**/
  log_stream.open(solutionpath + "/main"+ std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +".log", std::ios::binary);
  deallog.attach(log_stream);

}

static Parameters GetParameters() {
	ParameterHandler prm;
	ParameterReader param(prm);
	// input_file_name = "/home/pascal/workspace/waveguideproblem/Parameters/Parameters.xml";
	param.read_parameters(input_file_name);
	struct Parameters ret;
	prm.enter_subsection("Output");
	{
	  prm.enter_subsection("Optimization");
	  {
      prm.enter_subsection("Gnuplot");
      {
        ret.O_O_G_HistoryLive = prm.get_bool("Optimization History Live");
        ret.O_O_G_HistoryShapes = prm.get_bool("Optimization History Shapes");
        ret.O_O_G_History = prm.get_bool("Optimization History");
      }
      prm.leave_subsection();

      prm.enter_subsection("VTK");
      {
        prm.enter_subsection("TransformationWeights");
        {
            ret.O_O_V_T_TransformationWeightsAll = prm.get_bool("TransformationWeightsAll");
            ret.O_O_V_T_TransformationWeightsFirst = prm.get_bool("TransformationWeightsFirst");
            ret.O_O_V_T_TransformationWeightsLast = prm.get_bool("TransformationWeightsLast");
        }
        prm.leave_subsection();

        prm.enter_subsection("Solution");
        {
          ret.O_O_V_S_SolutionAll = prm.get_bool("SolutionAll");
          ret.O_O_V_S_SolutionFirst = prm.get_bool("SolutionFirst");
          ret.O_O_V_S_SolutionLast = prm.get_bool("SolutionLast");
        }
        prm.leave_subsection();

      }
      prm.leave_subsection();
	  }
	  prm.leave_subsection();

    prm.enter_subsection("Convergence");
    {
      prm.enter_subsection("DataFiles");
      {
        ret.O_C_D_ConvergenceFirst = prm.get_bool("ConvergenceFirst");
        ret.O_C_D_ConvergenceLast = prm.get_bool("ConvergenceLast");
        ret.O_C_D_ConvergenceAll = prm.get_bool("ConvergenceAll");
      }
      prm.leave_subsection();

      prm.enter_subsection("Plots");
      {
        ret.O_C_P_ConvergenceFirst = prm.get_bool("ConvergenceFirst");
        ret.O_C_P_ConvergenceLast = prm.get_bool("ConvergenceLast");
        ret.O_C_P_ConvergenceAll = prm.get_bool("ConvergenceAll");
      }
      prm.leave_subsection();

    }
    prm.leave_subsection();

    prm.enter_subsection("General");
    {
      ret.O_G_Summary = prm.get_bool("SummaryFile");
      ret.O_G_Log = prm.get_bool("LogFile");
    }
    prm.leave_subsection();

	}
  prm.leave_subsection();

  prm.enter_subsection("Measures");
  {
    prm.enter_subsection("Connectors");
    {
      std::string temp = prm.get("Shape");
      if(temp == "Circle") {
        ret.M_C_Shape = ConnectorType::Circle;
      } else {
        ret.M_C_Shape = ConnectorType::Rectangle;
      }
      ret.M_C_Dim1In = prm.get_double("Dimension1 In");
      ret.M_C_Dim2In = prm.get_double("Dimension2 In");
      ret.M_C_Dim1Out = prm.get_double("Dimension1 Out");
      ret.M_C_Dim2Out = prm.get_double("Dimension2 Out");
    }
    prm.leave_subsection();

    prm.enter_subsection("Region");
    {
      ret.M_R_XLength = prm.get_double("XLength");
      ret.M_R_YLength = prm.get_double("YLength");
      ret.M_R_ZLength = prm.get_double("ZLength");
    }
    prm.leave_subsection();

    prm.enter_subsection("Waveguide");
    {
      ret.M_W_Delta = prm.get_double("Delta");
      ret.M_W_epsilonin = prm.get_double("epsilon in");
      ret.M_W_epsilonout = prm.get_double("epsilon out");
      ret.M_W_Lambda = prm.get_double("Lambda");
      ret.M_W_Sectors = prm.get_integer("Sectors");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary Conditions");
    {
      std::string temp = prm.get("Type");
      if (temp == "PML") {
        ret.M_BC_Type= BoundaryConditionType::PML;
      } else {
        ret.M_BC_Type= BoundaryConditionType::HSIE;
      }
      ret.M_BC_Zplus = prm.get_integer("ZPlus");
      ret.M_BC_XMinus = prm.get_double("XMinus");
      ret.M_BC_XPlus = prm.get_double("XPlus");
      ret.M_BC_YMinus = prm.get_double("YMinus");
      ret.M_BC_YPlus = prm.get_double("YPlus");
      ret.M_BC_KappaXMax = prm.get_double("KappaXMax");
      ret.M_BC_KappaYMax = prm.get_double("KappaYMax");
      ret.M_BC_KappaZMax = prm.get_double("KappaZMax");
      ret.M_BC_SigmaXMax = prm.get_double("SigmaXMax");
      ret.M_BC_SigmaYMax = prm.get_double("SigmaYMax");
      ret.M_BC_SigmaZMax = prm.get_double("SigmaZMax");
      ret.M_BC_DampeningExponent = prm.get_double("DampeningExponentM");
    }
    prm.leave_subsection();

  }
  prm.leave_subsection();

  prm.enter_subsection("Schema");
  {
    ret.Sc_Homogeneity = prm.get_bool("Homogeneity");
    std::string temp = prm.get("Optimization Schema");
    if(temp == "Adjoint") {
      ret.Sc_Schema = OptimizationSchema::Adjoint;
    } else if (temp == "FD") {
      ret.Sc_Schema = OptimizationSchema::FD;
    }
    ret.Sc_OptimizationSteps = prm.get_integer("Optimization Steps");
    temp = prm.get("Stepping Method");
    if(temp == "Steepest") {
      ret.Sc_SteppingMethod = SteppingMethod::Steepest;
    } else if (temp == "CG") {
      ret.Sc_SteppingMethod = SteppingMethod::CG;
    } else if (temp == "LineSearch") {
      ret.Sc_SteppingMethod = SteppingMethod::LineSearch;
    }
  }
  prm.leave_subsection();

  prm.enter_subsection("Solver");
  {
    std::string temp = prm.get("Solver");
    if(temp == "GMRES"){
      ret.So_Solver = SolverOptions::GMRES;
    } else if (temp == "MINRES"){
      ret.So_Solver = SolverOptions::MINRES;
    } else if (temp == "UMFPACK") {
      ret.So_Solver = SolverOptions::UMFPACK;
    }
    ret.So_RestartSteps = prm.get_integer("GMRESSteps");
    temp = prm.get("Preconditioner");
    if (temp == "Sweeping") {
      ret.So_Preconditioner = PreconditionerOptions::Sweeping;
    } else if(temp == "Amesos_Lapack") {
      ret.So_Preconditioner = PreconditionerOptions::Amesos_Lapack;
    }
    ret.So_TotalSteps = prm.get_integer("Steps");
    ret.So_Precision = prm.get_double("Precision");
  }
  prm.leave_subsection();

  prm.enter_subsection("Constants");
  {
    ret.C_AllOne = prm.get_bool("AllOne");
    if(ret.C_AllOne) {
      ret.C_Epsilon = 1.0;
      ret.C_Mu = 1.0;
    } else {
      ret.C_Epsilon = prm.get_double("EpsilonZero");
      ret.C_Mu = prm.get_double("MuZero");
    }
    ret.C_c = 1.0 / sqrt(ret.C_Epsilon * ret.C_Mu);
    ret.C_f0 = ret.C_c/ret.M_W_Lambda;
    ret.C_Pi = prm.get_double("Pi");
    ret.C_k0 = 2.0 * ret.C_Pi / ret.M_W_Lambda;
    ret.C_omega = 2.0 * ret.C_Pi * ret.C_f0;
  }
  prm.leave_subsection();

  prm.enter_subsection("Refinement");
  {
    ret.R_Global = prm.get_integer("Global");
    ret.R_Local = prm.get_integer("SemiGlobal");
    ret.R_Interior = prm.get_integer("Internal");
  }
  prm.leave_subsection();

	ret.MPIC_World = MPI_COMM_WORLD;
	ret.MPI_Rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
	ret.NumberProcesses = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

	ret.Head = (ret.MPI_Rank == 0);

	if((int)ret.MPI_Rank > ret.NumberProcesses - ret.M_BC_Zplus -1 ) {
	  ret.PMLLayer = true;
	} else {
	  ret.PMLLayer = false;
	}

	ret.LayersPerSector = (ret.NumberProcesses - ret.M_BC_Zplus)/ret.M_W_Sectors;

	ret.LayerThickness = ret.M_R_ZLength / (ret.NumberProcesses - ret.M_BC_Zplus);

	ret.SectorThickness = ret.M_R_ZLength / ret.M_W_Sectors;

	ret.SystemLength = ret.NumberProcesses * ret.LayerThickness;

	ret.Maximum_Z = - (ret.M_R_ZLength/2.0) + ret.SystemLength;

	deallog.push("Checking Waveguide Properties");

	ret.Phys_V = 2 * ret.C_Pi * ret.M_C_Dim1In / ret.M_W_Lambda *std::sqrt(ret.M_W_epsilonin * ret.M_W_epsilonin - ret.M_W_epsilonout * ret.M_W_epsilonout);

	ret.So_ElementOrder = 0;

	deallog << "Normalized Frequency V: " << ret.Phys_V <<std::endl;

	if(ret.Phys_V> 1.5 && ret.Phys_V < 2.405){
	  deallog << "This Waveguide is Single Moded" <<std::endl;
	} else {
	  deallog << "This Waveguide is not Single Moded" <<std::endl;
	  double temp = ret.Phys_V * ret.M_W_Lambda;
	  deallog << "Minimum Lambda: " << temp/1.5 <<std::endl;
	  deallog << "Maximum Lambda: " << temp/2.405 <<std::endl;
	  deallog << "Current Lambda: " << ret.M_W_Lambda <<std::endl;
	}

	ret.Phys_SpotRadius = (0.65 + 1.619/(std::pow(ret.Phys_V, 1.5)) + 2.879/(std::pow(ret.Phys_V, 6))) * ret.M_C_Dim1In;

	deallog << "Spot Radius omega: " << ret.Phys_SpotRadius <<std::endl;

	deallog.pop();
	return ret;
}

inline double InterpolationPolynomial(double in_z, double in_val_zero, double in_val_one, double in_derivative_zero, double in_derivative_one) {
	if (in_z < 0.0) return in_val_zero;
	if (in_z > 1.0) return in_val_one;
	return (2*(in_val_zero - in_val_one) + in_derivative_zero + in_derivative_one) * pow(in_z,3) + (3*(in_val_one - in_val_zero) - (2*in_derivative_zero) - in_derivative_one)*pow(in_z,2) + in_derivative_zero*in_z + in_val_zero;
}

inline double InterpolationPolynomialDerivative(double in_z, double in_val_zero, double in_val_one, double in_derivative_zero, double in_derivative_one) {
	if (in_z < 0.0) return in_derivative_zero;
	if (in_z > 1.0) return in_derivative_one;
	return 3* (2*(in_val_zero - in_val_one) + in_derivative_zero + in_derivative_one) * pow(in_z,2) + 2*(3*(in_val_one - in_val_zero) - (2*in_derivative_zero) - in_derivative_one)*in_z + in_derivative_zero;
}

inline double InterpolationPolynomialZeroDerivative(double in_z , double in_val_zero, double in_val_one) {
	return InterpolationPolynomial(in_z, in_val_zero, in_val_one, 0.0, 0.0);
}

static double Distance2D (Point<3, double> position, Point<3, double> to = Point<3, double>()) {
		return sqrt((position(0)-to(0))*(position(0)-to(0)) + (position(1)-to(1))*(position(1)-to(1)));
}

inline Tensor<1, 3 , double> crossproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	Tensor<1,3,double> ret;
	ret[0] = a[1] * b[2] - a[2] * b[1];
	ret[1] = a[2] * b[0] - a[0] * b[2];
	ret[2] = a[0] * b[1] - a[1] * b[0];
	return ret;
}

inline double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<int dim> static void mesh_info(const parallel::distributed::Triangulation<dim> &tria, const std::string &filename)
{
	std::cout << "Mesh info:" << std::endl << " dimension: " << dim << std::endl << " no. of cells: " << tria.n_active_cells() << std::endl;
	{
		std::map<unsigned int, unsigned int> boundary_count;
		typename parallel::distributed::Triangulation<dim>::active_cell_iterator
		cell = tria.begin_active(),
		endc = tria.end();
		for (; cell!=endc; ++cell)
		{
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
			{
				if (cell->face(face)->at_boundary())
					boundary_count[cell->face(face)->boundary_id()]++;
			}
		}
		std::cout << " boundary indicators: ";
		for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
				it!=boundary_count.end();
				++it)
		{
			std::cout << it->first << "(" << it->second << " times) ";
		}
		std::cout << std::endl;
	}
	std::ofstream out (filename.c_str());
	GridOut grid_out;
	grid_out.write_vtk (tria, out);
	out.close();
	std::cout << " written to " << filename << std::endl << std::endl;
}

template<int dim> static void mesh_info(const parallel::distributed::Triangulation<dim> &tria)
{
  std::cout << "Mesh info:" << std::endl << " dimension: " << dim << std::endl << " no. of cells: " << tria.n_active_cells() << std::endl;
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    for (; cell!=endc; ++cell)
    {
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary())
          boundary_count[cell->face(face)->boundary_id()]++;
      }
    }
    std::cout << " boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
        it!=boundary_count.end();
        ++it)
    {
      std::cout << it->first << "(" << it->second << " times) ";
    }
    std::cout << std::endl;
  }

}

static double sigma (double in_z, double min, double max) {
	if( min == max ) return (in_z < min )? 0.0 : 1.0;
	if(in_z < min) return 0.0;
	if(in_z > max) return 1.0;
	double ret = 0;
	ret = (in_z - min) / ( max - min);
	if(ret < 0.0) ret = 0.0;
	if(ret > 1.0) ret = 1.0;
	return ret;
}

static Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= GlobalParams.M_R_XLength / 2.0 ;
  return q;
}

static Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= GlobalParams.M_R_YLength / 2.0 ;
  return q;
}

static Point<3> Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  double total_length = GlobalParams.SystemLength;
  q[2] *= total_length / 2.0;
  return q;
}

static Point<3> Triangulation_Shift_Z (const Point<3> &p)
{
  Point<3> q = p;
  q[2] += (GlobalParams.SystemLength - GlobalParams.M_R_ZLength)/2.0;
  return q;
}

static Point<3> Triangulation_Stretch_to_circle (const Point<3> &p)
{
	Point<3> q = p;
	if(abs(q[0]) < 0.01 && abs(q[1]) - 0.25 < 0.01 ) {
		q[1] *= sqrt(2);
	}
	if(abs(q[1]) < 0.01 && abs(q[0]) - 0.25 < 0.01 ) {
			q[0] *= sqrt(2);
	}
	return q;
}

static Point<3> Triangulation_Stretch_Computational_Radius (const Point<3> &p)
{
	double r_goal = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;
	//double r_current = (GlobalParams.PRM_M_R_XLength ) / 7.12644;
	double r_current = (GlobalParams.M_R_XLength ) / 5.65;
	double r_max = (GlobalParams.M_R_XLength / 2.0 ) * (1.0 - (2.0*GlobalParams.M_BC_XMinus));
	double r_point = sqrt(p[0]*p[0] + p[1]*p[1]);
	double factor = InterpolationPolynomialZeroDerivative(sigma(r_point, r_current, r_max), r_goal/r_current , 1.0);
	Point<3> q = p;
	q[0] *= factor;
	q[1] *= factor;
	return q;
}

static Point<3> Triangulation_Stretch_Computational_Rectangle (const Point<3> &p)
{
  double x_goal = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;
  double x_current = (GlobalParams.M_R_XLength ) / 6;
  double x_max = GlobalParams.M_R_XLength / 2.0  - GlobalParams.M_BC_XMinus ;
  double x_point = abs(p[0]);
  double factor = InterpolationPolynomialZeroDerivative(sigma(x_point, x_current, x_max), x_goal/x_current , 1.0);
  Point<3> q = p;
  q[0] *= factor;
  double y_goal = (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out)/2.0;
  double y_current = (GlobalParams.M_R_YLength ) / 6;
  double y_max = GlobalParams.M_R_YLength / 2.0 - GlobalParams.M_BC_YMinus;
  double y_point = abs(p[1]);
  factor = InterpolationPolynomialZeroDerivative(sigma(y_point, y_current, y_max), y_goal/y_current , 1.0);
  q[1] *= factor;
  return q;
}

static bool System_Coordinate_in_Waveguide(Point<3> p){
	double value = Distance2D(p);
	return ( value < (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0);
}

static double TEMode00 (Point<3, double> p ,const unsigned int component)
{

	if(component == 0) {
		// double d2 = (2* Distance2D(p)) / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) ;
	  double d2 = Distance2D(p);
		// return exp(-d2*d2 / (GlobalParams.Phys_SpotRadius * GlobalParams.Phys_SpotRadius));
	  return exp(-d2*d2 / 2.25);
	  //return 1.0;
	}
	return 0.0;
}

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#endif
