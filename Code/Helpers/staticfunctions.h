#pragma once

#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <fstream>
#include "../SpaceTransformations/SpaceTransformation.h"
#include "./Parameters.h"
#include "../Core/Types.h"

extern std::string solutionpath;
extern std::ofstream log_stream;
extern std::string constraints_filename;
extern std::string assemble_filename;
extern std::string precondition_filename;
extern std::string solver_filename;
extern std::string total_filename;
extern int StepsR;
extern int StepsPhi;
extern int alert_counter;
extern std::string input_file_name;
extern SpaceTransformation *the_st;

/**
 * For given vectors \f$\boldsymbol{a},\boldsymbol{b} \in \mathbb{R}^3\f$, this
 * function calculates the following crossproduct: \f[\boldsymbol{a} \ times
 * \boldsymbol{b} = \begin{pmatrix} a_2 b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\
 * a_1b_2 - a_2b_1\end{pmatrix}\f]
 */
Tensor<1, 3, double> crossproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

ComplexNumber matrixD(int in_row, int in_column,
                             ComplexNumber in_k0);

bool compareDofBaseData(std::pair<DofNumber, Position> c1,
    std::pair<DofNumber, Position> c2);

bool compareDofBaseDataAndOrientation(
    DofIndexAndOrientationAndPosition,
    DofIndexAndOrientationAndPosition);

bool areDofsClose(const DofIndexAndOrientationAndPosition &a,
    const DofIndexAndOrientationAndPosition &b);

void set_the_st(SpaceTransformation *);

double dotproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

template <int dim>
void mesh_info(const Triangulation<dim>, const std::string);

template <int dim>
void mesh_info(const Triangulation<dim>);

Parameters GetParameters();

Position Triangulation_Shit_To_Local_Geometry(
    const Position &p);

Position Transform_4_to_5(const Position &p);

Position Transform_3_to_5(const Position &p);

Position Transform_2_to_5(const Position &p);

Position Transform_1_to_5(const Position &p);

Position Transform_0_to_5(const Position &p);

Position Transform_5_to_4(const Position &p);

Position Transform_5_to_3(const Position &p);

Position Transform_5_to_2(const Position &p);

Position Transform_5_to_1(const Position &p);

Position Transform_5_to_0(const Position &p);

inline bool file_exists(const std::string &name);

void PrepareStreams();

double Distance2D(Position, Position = Position());

std::vector<types::global_dof_index> Add_Zero_Restraint(
    AffineConstraints<double> *, DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, IndexSet);

void add_vector_of_indices(IndexSet *, std::vector<types::global_dof_index>);

double hmax_for_cell_center(Position);

double InterpolationPolynomial(double, double, double, double, double);

double InterpolationPolynomialDerivative(double, double, double, double,
                                         double);

double InterpolationPolynomialZeroDerivative(double, double, double);

double sigma(double, double, double);

auto compute_center_of_triangulation(const Mesh*) -> Position;

bool get_orientation(
    const Position &vertex_1,
    const Position &vertex_2);

NumericVectorLocal crossproduct(const NumericVectorLocal &u,const NumericVectorLocal &v);

Position crossproduct(const Position &u,const Position &v);

void multiply_in_place(const ComplexNumber factor_1, NumericVectorLocal &factor_2);

void print_info(const std::string &label, const std::string &message, bool blocking = false, LoggingLevel level = LoggingLevel::DEBUG_ALL);
void print_info(const std::string &label, const unsigned int message, bool blocking = false, LoggingLevel level = LoggingLevel::DEBUG_ALL);
void print_info(const std::string &label, const std::vector<unsigned int> &message, bool blocking = false, LoggingLevel level = LoggingLevel::DEBUG_ALL);
void print_info(const std::string &label, const std::array<bool,6> &message, bool blocking = false, LoggingLevel level = LoggingLevel::DEBUG_ALL);

bool is_visible_message_in_current_logging_level(LoggingLevel level = LoggingLevel::DEBUG_ALL);
void write_print_message(const std::string &label, const std::string &message);