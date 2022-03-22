#pragma once

/**
 * @file staticfunctions.h
 * @author your name (you@domain.com)
 * @brief This is an important file since it contains all the utility functions used anywhere in the code.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <fstream>
#include "./Parameters.h"
#include "./ParameterOverride.h"
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

using namespace dealii;

/**
 * For given vectors \f$\boldsymbol{a},\boldsymbol{b} \in \mathbb{R}^3\f$, this
 * function calculates the following crossproduct: \f[\boldsymbol{a} \ times
 * \boldsymbol{b} = \begin{pmatrix} a_2 b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\
 * a_1b_2 - a_2b_1\end{pmatrix}\f]
 */
Tensor<1, 3, double> crossproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

std::string exec(const char* cmd);

ComplexNumber matrixD(int in_row, int in_column, ComplexNumber in_k0);

std::pair<DofNumber, DofNumber> get_max_and_min_dof_for_interface_data(std::vector<InterfaceDofData> in_data);

bool comparePositions(Position p1, Position p2);

bool compareDofBaseData(std::pair<DofNumber, Position> c1, std::pair<DofNumber, Position> c2);

bool compareDofBaseDataAndOrientation(InterfaceDofData, InterfaceDofData);

bool compareSurfaceCellData(SurfaceCellData c1, SurfaceCellData c2);

bool compareDofDataByGlobalIndex(InterfaceDofData, InterfaceDofData);

bool areDofsClose(const InterfaceDofData &a, const InterfaceDofData &b);

bool compareFEAdjointEvals(const FEAdjointEvaluation field_a,const FEAdjointEvaluation field_b);

double dotproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

void mesh_info(Triangulation<3> *, std::string);

template <int dim>
void mesh_info(const Triangulation<dim>);

Parameters GetParameters(const std::string run_file,const std::string case_file, ParameterOverride& in_po);

Position Triangulation_Shit_To_Local_Geometry(const Position &p);

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

double Distance2D(const Position &, const Position & = Position());

double Distance3D(const Position &, const Position & = Position());

std::vector<types::global_dof_index> Add_Zero_Restraint(
    AffineConstraints<double> *, DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, IndexSet);

void add_vector_of_indices(IndexSet *, std::vector<types::global_dof_index>);

double hmax_for_cell_center(Position);

double InterpolationPolynomial(double, double, double, double, double);

double InterpolationPolynomialDerivative(double, double, double, double, double);

double InterpolationPolynomialZeroDerivative(double, double, double);

double sigma(double, double, double);

auto compute_center_of_triangulation(const Mesh*) -> Position;

bool get_orientation(const Position &vertex_1, const Position &vertex_2);

NumericVectorLocal crossproduct(const NumericVectorLocal &u,const NumericVectorLocal &v);

Position crossproduct(const Position &u,const Position &v);

void multiply_in_place(const ComplexNumber factor_1, NumericVectorLocal &factor_2);

void print_info(const std::string &label, const std::string &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
void print_info(const std::string &label, const unsigned int message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
void print_info(const std::string &label, const std::vector<unsigned int> &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
void print_info(const std::string &label, const std::array<bool,6> &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);

bool is_visible_message_in_current_logging_level(LoggingLevel level = LoggingLevel::DEBUG_ONE);
void write_print_message(const std::string &label, const std::string &message);

BoundaryId opposing_Boundary_Id(BoundaryId b_id);

bool are_opposing_sites(BoundaryId a, BoundaryId b);

std::vector<DofCouplingInformation> get_coupling_information(std::vector<InterfaceDofData> &dofs_interface_1, std::vector<InterfaceDofData> &dofs_interface_2);

Position deal_vector_to_position(NumericVectorLocal &inp);

auto get_affine_constraints_for_InterfaceData(std::vector<InterfaceDofData> &dofs_interface_1, std::vector<InterfaceDofData> &dofs_interface_2, const unsigned int max_dof) -> Constraints;

void shift_interface_dof_data(std::vector<InterfaceDofData> * dofs_interface_1, unsigned int shift);

dealii::Triangulation<3> reforge_triangulation(dealii::Triangulation<3> * original_triangulation);

ComplexNumber conjugate(const ComplexNumber & in_number);

bool is_absorbing_boundary(SurfaceType in_st);

double norm_squared(const ComplexNumber in_c);

bool are_edge_dofs_locally_owned(BoundaryId self, BoundaryId other, unsigned int in_level);

std::vector<BoundaryId> get_adjacent_boundary_ids(BoundaryId self);

SweepingDirection get_sweeping_direction_for_level(unsigned int in_level);

int generate_tag(unsigned int global_rank_sender, unsigned int receiver, unsigned int level);

std::vector<std::string> split(std::string str, std::string token);

SolverOptions solver_option(std::string in_name);

std::vector<double> fe_evals_to_double(const std::vector<FEAdjointEvaluation>& inp);

std::vector<FEAdjointEvaluation> fe_evals_from_double(const std::vector<double>& inp);

Position adjoint_position_transformation(const Position in_p);

dealii::Tensor<1,3,ComplexNumber> adjoint_field_transformation(const dealii::Tensor<1,3,ComplexNumber> in_field);