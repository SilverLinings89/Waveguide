#include "staticfunctions.h"
#include <deal.II/base/point.h>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <limits>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/dofs/dof_handler.h>
#include "GeometryManager.h"
#include "ParameterReader.h"
#include "Parameters.h"
#include "ShapeDescription.h"

using namespace dealii;

#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif  /* _COLORS_ */

extern Parameters GlobalParams;
extern GeometryManager Geometry;
unsigned int message_count = 0;

void set_the_st(SpaceTransformation *in_st) { the_st = in_st; }

auto compute_center_of_triangulation(const Mesh *in_mesh) -> Position {
  double x_average = 0;
  double y_average = 0;
  double z_average = 0;
  const unsigned int n_vertices = in_mesh->n_vertices();
  for (auto &cell : in_mesh->active_cell_iterators()) {
    for (unsigned int i = 0; i < 8; i++) {
      Position v = cell->vertex(i);
      x_average += v[0] / n_vertices;
      y_average += v[1] / n_vertices;
      z_average += v[2] / n_vertices;
    }
  }
  return {x_average, y_average, z_average};
}

bool compareDofBaseData(std::pair<DofNumber, Position> c1,
    std::pair<DofNumber, Position> c2) {
  if (c1.second[2] != c2.second[2]) {
    return c1.second[2] < c2.second[2];
  }
  if (c1.second[1] != c2.second[1]) {
    return c1.second[1] < c2.second[1];
  }
  return c1.second[0] < c2.second[0];
}

bool areDofsClose(const DofIndexAndOrientationAndPosition &a,
    const DofIndexAndOrientationAndPosition &b) {
  double dist = 0;
  for (unsigned int i = 0; i < 3; i++) {
    dist += (a.position[i] - b.position[i]) * (a.position[i] - b.position[i]);
  }
  return std::sqrt(dist) < 0.001;
}

bool compareDofBaseDataAndOrientation(DofIndexAndOrientationAndPosition c1, DofIndexAndOrientationAndPosition c2) {
  if (c1.position[2] != c2.position[2]) {
    return c1.position[2] < c2.position[2];
  }
  if (c1.position[1] != c2.position[1]) {
    return c1.position[1] < c2.position[1];
  }
  return c1.position[0] < c2.position[0];
}

bool compareDofDataByGlobalIndex(DofIndexAndOrientationAndPosition c1, DofIndexAndOrientationAndPosition c2) {
  return c1.index < c2.index;
}

void alert() {
  MPI_Barrier(MPI_COMM_WORLD);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    std::cout << "Alert: " << alert_counter << std::endl;
  }
  alert_counter++;
}

ComplexNumber matrixD(int in_row, int in_column,
                             ComplexNumber in_k0) {
  ComplexNumber ret(0, 0);
  if (std::abs(in_row - in_column) > 1) {
    return ret;
  }
  ComplexNumber part = 1.0 / (ComplexNumber(0, 2) * in_k0);
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
    ret += ComplexNumber(1, 0) - part;
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

Parameters GetParameters(std::string run_filename, std::string case_filename) {
  ParameterReader pr;
  pr.declare_parameters();
  
  struct Parameters ret = pr.read_parameters(run_filename, case_filename);
  
  ret.MPI_Rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ret.NumberProcesses = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  ret.complete_data();
  return ret;
}

double Distance2D(Position position, Position to) {
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
  print_info("mesh_info", "Mesh info:\ndimension: " + std::to_string(dim) + "\nno. of cells: " + std::to_string(tria.n_active_cells()), false, LoggingLevel::PRODUCTION_ALL);
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
    std::string m = " boundary indicators: ";
    for (auto &it : boundary_count) {
      m += std::to_string(it.first) + "(" + std::to_string(it.second) + " times) ";
    }
    print_info("mesh_info",m, false, LoggingLevel::PRODUCTION_ALL);
  }
  std::ofstream out(filename.c_str());
  GridOut grid_out;
  grid_out.write_vtk(tria, out);
  out.close();
  print_info("mesh_info" , "written to " + filename, false, LoggingLevel::DEBUG_ONE);
}

template <int dim>
void mesh_info(const Triangulation<dim> &tria) {
  print_info("mesh_info", "Mesh info:\ndimension: " + std::to_string(dim) + "\nno. of cells: " + std::to_string(tria.n_active_cells()), false, LoggingLevel::PRODUCTION_ALL);
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
    std::string m = " boundary indicators: ";
    for (auto &it : boundary_count) {
      m += std::to_string(it.first) + "(" + std::to_string(it.second) + " times) ";
    }
    print_info("mesh_info",m, false, LoggingLevel::PRODUCTION_ALL);
  }
}

Position Triangulation_Shit_To_Local_Geometry(
    const Position &p) {
  Position q = p;

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

Position Transform_4_to_5(const Position &p) {
  Position q = p;
  q[0] = -p[0];
  return q;
}

Position Transform_3_to_5(const Position &p) {
  Position q = p;
  q[0] = p[0];
  q[1] = -p[2];
  q[2] = p[1];
  return q;
}

Position Transform_2_to_5(const Position &p) {
  Position q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = p[1];
  return q;
}

Position Transform_1_to_5(const Position &p) {
  Position q = p;
  q[0] = -p[2];
  q[2] = p[0];
  return q;
}

Position Transform_0_to_5(const Position &p) {
  Position q = p;
  q[0] = p[2];
  q[2] = p[0];
  return q;
}

Position Transform_5_to_4(const Position &p) {
  Position q = p;
  q[0] = -p[0];
  return q;
}

Position Transform_5_to_3(const Position &p) {
  Position q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = -p[1];
  return q;
}

Position Transform_5_to_2(const Position &p) {
  Position q = p;
  q[0] = p[0];
  q[1] = p[2];
  q[2] = p[1];
  return q;
}

Position Transform_5_to_1(const Position &p) {
  Position q = p;
  q[0] = p[2];
  q[2] = -p[0];
  return q;
}

Position Transform_5_to_0(const Position &p) {
  Position q = p;
  q[0] = p[2];
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

double hmax_for_cell_center(Position in_center) {
  double h_max_in = 0.05;
  double h_max_out = 0.1;
  return (std::abs(in_center[0]) < GlobalParams.Width_of_waveguide / 2.0 &&
          std::abs(in_center[0]) < GlobalParams.Height_of_waveguide / 2.0)
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

bool get_orientation(
    const Position &vertex_1,
    const Position &vertex_2) {
  double abs_max = -1.0;
  unsigned int max_component = 0;
  for(unsigned int i = 0; i < 3; i++) {
    const double abs = std::abs(vertex_2[i] - vertex_1[1]);
    if( abs > abs_max) {
        max_component = i;
        abs_max = abs;
    }
  }
  return (vertex_2[max_component] - vertex_1[max_component]) > 0;
}

NumericVectorLocal crossproduct(const NumericVectorLocal &u,const NumericVectorLocal &v) {
  NumericVectorLocal ret(3);
  ret[0]=(u[1]*v[2]- u[2]*v[1]);
  ret[1]=(u[2]*v[0]- u[0]*v[2]);
  ret[2]=(u[0]*v[1]- u[1]*v[0]);
  return ret;
}

Position crossproduct(const Position &u,const Position &v) {
  Position ret;
  ret[0]=(u[1]*v[2]- u[2]*v[1]);
  ret[1]=(u[2]*v[0]- u[0]*v[2]);
  ret[2]=(u[0]*v[1]- u[1]*v[0]);
  return ret;
}

void multiply_in_place(const ComplexNumber factor_1, NumericVectorLocal &factor_2) {
  for(unsigned int i = 0; i < factor_2.size(); i++) {
    factor_2[i] *= factor_1;
  }
}

void print_info(const std::string &label, const std::string &message, bool blocking, LoggingLevel logging_level) {
  if(blocking) MPI_Barrier(MPI_COMM_WORLD);
  if(is_visible_message_in_current_logging_level(logging_level)) {
    write_print_message(label, message);
  }
  if(blocking) MPI_Barrier(MPI_COMM_WORLD);
}

void print_info(const std::string &label, const unsigned int message, bool blocking, LoggingLevel logging_level) {
  print_info(label, std::to_string(message), blocking, logging_level);
}

void print_info(const std::string &label, const std::vector<unsigned int> &message, bool blocking, LoggingLevel logging_level) {
  std::string message_string = "";
  for(unsigned int i = 0; i < message.size(); i++) message_string += std::to_string(message[i]) + " ";
  print_info(label, message_string, blocking, logging_level);
}

void print_info(const std::string &label, const std::array<bool,6> &message, bool blocking, LoggingLevel logging_level) {
  std::string m = "";
  for(unsigned int i = 0; i < message.size(); i++) {
    if(message[i]) {
      m += std::to_string(i) + " true ";
    } else {
      m += std::to_string(i) + " false ";
    }
  }
  print_info(label, m, blocking, logging_level);  
}

bool is_visible_message_in_current_logging_level(LoggingLevel level) {
  if(level == LoggingLevel::DEBUG_ONE || level == LoggingLevel::PRODUCTION_ONE) {
    if(GlobalParams.MPI_Rank != 0) return false;
  }
  bool admissible = level >= GlobalParams.Logging_Level;
  if(level == LoggingLevel::DEBUG_ONE || level == LoggingLevel::PRODUCTION_ONE) {
    admissible &= GlobalParams.MPI_Rank == 0;
  }
  return admissible;
}

void write_print_message(const std::string &label, const std::string &message) {
  std::cout << "[" << GlobalParams.MPI_Rank<< ":" << GlobalParams.Index_in_x_direction << "x" << GlobalParams.Index_in_y_direction << "x" << GlobalParams.Index_in_z_direction << "]" << "\x1B[1m\x1B[31m"  << label << "\x1B[0m\x1B[0m" << ": " << message << std::endl;
}

BoundaryId opposing_Boundary_Id(BoundaryId b_id) {
  if(b_id % 2  == 0) {
    return b_id + 1;
  } else {
    return b_id - 1;
  }
}

bool are_opposing_sites(BoundaryId a, BoundaryId b) {
  return a != b && a/2 == b/2;
}