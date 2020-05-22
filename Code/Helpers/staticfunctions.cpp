#ifndef StaticFunctionsFlag
#define StaticFunctionsFlag

#include "staticfunctions.h"
#include <deal.II/base/point.h>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include "GeometryManager.h"
#include "ParameterReader.h"
#include "Parameters.h"
#include "ShapeDescription.h"

using namespace dealii;

extern Parameters GlobalParams;
extern GeometryManager Geometry;

void set_the_st(SpaceTransformation *in_st) { the_st = in_st; }

bool compareDofBaseData(std::pair<int, Point<3, double>> c1,
    std::pair<int, Point<3, double>> c2) {
  if (c1.second[2] != c2.second[2]) {
    return c1.second[2] < c2.second[2];
  }
  if (c1.second[1] != c2.second[1]) {
    return c1.second[1] < c2.second[1];
  }

  return c1.second[0] < c2.second[0];
}


void alert() {
  MPI_Barrier(MPI_COMM_WORLD);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout << "Alert: " << alert_counter << std::endl;
  }
  alert_counter++;
}

std::complex<double> matrixD(int in_row, int in_column,
                             std::complex<double> in_k0) {
  std::complex<double> ret(0, 0);
  if (std::abs(in_row - in_column) > 1) {
    return ret;
  }
  std::complex<double> part = 1.0 / (std::complex<double>(0, 2) * in_k0);
  if (in_row == 0) {
    if (in_column == 0) {
      return (-1.0 * part) + 1.0;
    } else {
      return part;
    }
  } else {
    if (in_column == in_row - 1) {
      ret = (double)in_row * part * 1.0;
    }
    if (in_column == in_row) {
      ret = (double)in_row * part * (-2.0);
    }
    if (in_column == in_row + 1) {
      ret = (double)in_row * part * 1.0;
    }
  }
  if (in_column == in_row) {
    ret += std::complex<double>(1, 0) - part;
  }
  if (in_column == in_row + 1) {
    ret += part;
  }
  return ret;
}

void PrepareStreams() {
  char *pPath;
  pPath = getenv("WORK");
  bool seperate_solutions = (pPath != NULL);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    deallog.depth_console(10);
  } else {
    deallog.depth_console(0);
  }
  int i = 0;
  bool dir_exists = true;
  while (dir_exists) {
    std::stringstream out;
    if (seperate_solutions) {
      out << pPath << "/";
    }
    out << "Solutions/run";
    out << i;
    solutionpath = out.str();
    struct stat myStat;
    const char *myDir = solutionpath.c_str();
    if ((stat(myDir, &myStat) == 0) &&
        (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
      i++;
    } else {
      dir_exists = false;
    }
  }
  i = Utilities::MPI::max(i, MPI_COMM_WORLD);
  std::stringstream out;
  if (seperate_solutions) {
    out << pPath << "/";
  }
  out << "Solutions/run";

  out << i;
  solutionpath = out.str();
  mkdir(solutionpath.c_str(), ACCESSPERMS);

  log_stream.open(
      solutionpath + "/main" +
          std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
          ".log",
      std::ios::binary);

  deallog.attach(log_stream);
}

Parameters GetParameters() {
  ParameterHandler prm;
  ParameterReader param(prm);
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
          ret.O_O_V_T_TransformationWeightsAll =
              prm.get_bool("TransformationWeightsAll");
          ret.O_O_V_T_TransformationWeightsFirst =
              prm.get_bool("TransformationWeightsFirst");
          ret.O_O_V_T_TransformationWeightsLast =
              prm.get_bool("TransformationWeightsLast");
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
    prm.enter_subsection("PredefinedCases");
    {
      ret.M_PC_Use = prm.get_bool("ComputeCase");
      ret.M_PC_Case = prm.get_integer("SelectCase");
    }
    prm.leave_subsection();
    prm.enter_subsection("Connectors");
    {
      std::string temp = prm.get("Shape");
      if (temp == "Circle") {
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
  }
  prm.leave_subsection();

  prm.enter_subsection("Schema");
  {
    ret.Sc_Homogeneity = prm.get_bool("Homogeneity");
    std::string temp = prm.get("Optimization Schema");
    if (temp == "Adjoint") {
      ret.Sc_Schema = OptimizationSchema::Adjoint;
    } else if (temp == "FD") {
      ret.Sc_Schema = OptimizationSchema::FD;
    }
    ret.Sc_OptimizationSteps = prm.get_integer("Optimization Steps");
    temp = prm.get("Stepping Method");
    if (temp == "Steepest") {
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
    ret.So_TotalSteps = prm.get_integer("Steps");
    ret.So_Precision = prm.get_double("Precision");
  }
  prm.leave_subsection();

  prm.enter_subsection("Constants");
  {
    ret.C_AllOne = prm.get_bool("AllOne");
    if (ret.C_AllOne) {
      ret.C_Epsilon = 1.0;
      ret.C_Mu = 1.0;
    } else {
      ret.C_Epsilon = prm.get_double("EpsilonZero");
      ret.C_Mu = prm.get_double("MuZero");
    }
    ret.C_c = 1.0 / sqrt(ret.C_Epsilon * ret.C_Mu);
    ret.C_f0 = ret.C_c / ret.M_W_Lambda;
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

  ret.MPI_Rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ret.NumberProcesses = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  deallog << "Case Detection: ";
  if (ret.M_PC_Use) {
    deallog << "Using case " << ret.M_PC_Case << std::endl;
    std::ifstream input("Modes/test.csv");
    std::string line;
    int counter = 0;
    bool case_found = false;
    if (ret.M_PC_Case >= 0 && ret.M_PC_Case < 57) {
      while (std::getline(input, line) && counter < 57) {
        if (counter == ret.M_PC_Case) {
          ret.sd.SetByString(line);
          case_found = true;
        }
        counter++;
      }
      if (!case_found) {
        deallog << "There was a severe error. The case was not found therefore "
                   "not initialized."
                << std::endl;
      } else {
        ret.M_W_Sectors = ret.sd.Sectors;
        ret.M_R_ZLength = ret.sd.z[ret.sd.Sectors - 1] - ret.sd.z[0];
      }
    }
  } else {
    deallog << "Not using case." << std::endl;
  }

  ret.SectorThickness = ret.M_R_ZLength / ret.M_W_Sectors;

  deallog.push("Checking Waveguide Properties");

  ret.Phys_V = 2 * ret.C_Pi * ret.M_C_Dim1In / ret.M_W_Lambda *
               std::sqrt(ret.M_W_epsilonin * ret.M_W_epsilonin -
                         ret.M_W_epsilonout * ret.M_W_epsilonout);

  ret.So_ElementOrder = 0;

  deallog << "Normalized Frequency V: " << ret.Phys_V << std::endl;

  if (ret.Phys_V > 1.5 && ret.Phys_V < 2.405) {
    deallog << "This Waveguide is Single Moded" << std::endl;
  } else {
    deallog << "This Waveguide is not Single Moded" << std::endl;
    double temp = ret.Phys_V * ret.M_W_Lambda;
    deallog << "Minimum Lambda: " << temp / 1.5 << std::endl;
    deallog << "Maximum Lambda: " << temp / 2.405 << std::endl;
    deallog << "Current Lambda: " << ret.M_W_Lambda << std::endl;
  }

  ret.Phys_SpotRadius = (0.65 + 1.619 / (std::pow(ret.Phys_V, 1.5)) +
                         2.879 / (std::pow(ret.Phys_V, 6))) *
                        ret.M_C_Dim1In;

  deallog << "Spot Radius omega: " << ret.Phys_SpotRadius << std::endl;

  // Computing block_location for this process:

  ret.Index_in_x_direction =
      ret.MPI_Rank % (ret.Blocks_in_y_direction * ret.Blocks_in_z_direction);
  ret.Index_in_z_direction =
      ret.MPI_Rank / (ret.Blocks_in_x_direction * ret.Blocks_in_y_direction);
  ret.Index_in_y_direction =
      (ret.MPI_Rank % (ret.Blocks_in_x_direction * ret.Blocks_in_y_direction)) /
      ret.Blocks_in_z_direction;
  deallog.pop();
  ret.HSIE_SWEEPING_LEVEL = 1;
  return ret;
}

double Distance2D(Point<3, double> position, Point<3, double> to) {
  return sqrt((position(0) - to(0)) * (position(0) - to(0)) +
              (position(1) - to(1)) * (position(1) - to(1)));
}

Tensor<1, 3, double> crossproduct(Tensor<1, 3, double> a,
                                  Tensor<1, 3, double> b) {
  Tensor<1, 3, double> ret;
  ret[0] = a[1] * b[2] - a[2] * b[1];
  ret[1] = a[2] * b[0] - a[0] * b[2];
  ret[2] = a[0] * b[1] - a[1] * b[0];
  return ret;
}

double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <int dim>
void mesh_info(const Triangulation<dim> &tria, const std::string &filename) {
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator cell =
        tria.begin_active();
    typename Triangulation<dim>::active_cell_iterator endc = tria.end();
    for (; cell != endc; ++cell) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face) {
        if (cell->face(face)->at_boundary())
          boundary_count[cell->face(face)->boundary_id()]++;
      }
    }
    std::cout << " boundary indicators: ";
    for (auto &it : boundary_count) {
      std::cout << it.first << "(" << it.second << " times) ";
    }
    std::cout << std::endl;
  }
  std::ofstream out(filename.c_str());
  GridOut grid_out;
  grid_out.write_vtk(tria, out);
  out.close();
  std::cout << " written to " << filename << std::endl << std::endl;
}

template <int dim>
void mesh_info(const Triangulation<dim> &tria) {
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator cell =
        tria.begin_active();
    typename Triangulation<dim>::active_cell_iterator endc = tria.end();
    for (; cell != endc; ++cell) {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face) {
        if (cell->face(face)->at_boundary())
          boundary_count[cell->face(face)->boundary_id()]++;
      }
    }
    std::cout << " boundary indicators: ";
    for (auto &it : boundary_count) {
      std::cout << it.first << "(" << it.second << " times) ";
    }
    std::cout << std::endl;
  }
}

Point<3, double> Triangulation_Shit_To_Local_Geometry(
    const Point<3, double> &p) {
  Point<3, double> q = p;

  if (q[0] < 0) {
    q[0] = Geometry.local_x_range.first;
  } else {
    q[0] = Geometry.local_x_range.second;
  }
  if (q[1] < 0) {
    q[1] = Geometry.local_y_range.first;
  } else {
    q[1] = Geometry.local_y_range.second;
  }
  if (q[2] < 0) {
    q[3] = Geometry.local_z_range.first;
  } else {
    q[3] = Geometry.local_z_range.second;
  }
  return q;
}

Point<3, double> Transform_4_to_5(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = -p[0];
  return q;
}

Point<3, double> Transform_3_to_5(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[0];
  q[1] = -p[2];
  q[2] = p[1];
  return q;
}

Point<3, double> Transform_2_to_5(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = p[1];
  return q;
}

Point<3, double> Transform_1_to_5(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = -p[2];
  q[1] = p[1];
  q[2] = p[0];
  return q;
}

Point<3, double> Transform_0_to_5(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[2];
  q[1] = p[1];
  q[2] = p[0];
  return q;
}

Point<3, double> Transform_5_to_4(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = -p[0];
  return q;
}

Point<3, double> Transform_5_to_3(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = -p[1];
  return q;
}

Point<3, double> Transform_5_to_2(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = p[1];
  return q;
}

Point<3, double> Transform_5_to_1(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[2];
  q[1] = p[1];
  q[2] = -p[0];
  return q;
}

Point<3, double> Transform_5_to_0(const Point<3, double> &p) {
  Point<3, double> q = p;
  q[0] = p[2];
  q[1] = p[1];
  q[2] = p[0];
  return q;
}

inline bool file_exists(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

void add_vector_of_indices(dealii::IndexSet *in_index_set,
                           std::vector<types::global_dof_index> in_indices) {
  for (unsigned int i = 0; i < in_indices.size(); i++) {
    in_index_set->add_index(in_indices[i]);
  }
}

double hmax_for_cell_center(Point<3, double> in_center) {
  double h_max_in = 0.05;
  double h_max_out = 0.1;
  return (std::abs(in_center[0]) < GlobalParams.M_C_Dim1In / 2.0 &&
          std::abs(in_center[0]) < GlobalParams.M_C_Dim2In / 2.0)
             ? h_max_in
             : h_max_out;
}

double InterpolationPolynomial(double in_z, double in_val_zero,
                               double in_val_one, double in_derivative_zero,
                               double in_derivative_one) {
  if (in_z < 0.0) return in_val_zero;
  if (in_z > 1.0) return in_val_one;
  return (2 * (in_val_zero - in_val_one) + in_derivative_zero +
          in_derivative_one) *
             pow(in_z, 3) +
         (3 * (in_val_one - in_val_zero) - (2 * in_derivative_zero) -
          in_derivative_one) *
             pow(in_z, 2) +
         in_derivative_zero * in_z + in_val_zero;
}

double InterpolationPolynomialDerivative(double in_z, double in_val_zero,
                                         double in_val_one,
                                         double in_derivative_zero,
                                         double in_derivative_one) {
  if (in_z < 0.0) return in_derivative_zero;
  if (in_z > 1.0) return in_derivative_one;
  return 3 *
             (2 * (in_val_zero - in_val_one) + in_derivative_zero +
              in_derivative_one) *
             pow(in_z, 2) +
         2 *
             (3 * (in_val_one - in_val_zero) - (2 * in_derivative_zero) -
              in_derivative_one) *
             in_z +
         in_derivative_zero;
}

double InterpolationPolynomialZeroDerivative(double in_z, double in_val_zero,
                                             double in_val_one) {
  return InterpolationPolynomial(in_z, in_val_zero, in_val_one, 0.0, 0.0);
}

double sigma(double in_z, double min, double max) {
  if (min == max) return (in_z < min) ? 0.0 : 1.0;
  if (in_z < min) return 0.0;
  if (in_z > max) return 1.0;
  double ret = 0;
  ret = (in_z - min) / (max - min);
  if (ret < 0.0) ret = 0.0;
  if (ret > 1.0) ret = 1.0;
  return ret;
}

#endif
