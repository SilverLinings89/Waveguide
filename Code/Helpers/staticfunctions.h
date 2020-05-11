
#ifndef STATICFUNCTIONS_H_
#define STATICFUNCTIONS_H_

#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <fstream>
#include "../SpaceTransformations/SpaceTransformation.h"
#include "./Parameters.h"

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

std::complex<double> matrixD(int in_row, int in_column,
                             std::complex<double> in_k0);

void set_the_st(SpaceTransformation *);

double dotproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

template <int dim>
void mesh_info(const Triangulation<dim>, const std::string);

template <int dim>
void mesh_info(const Triangulation<dim>);

Parameters GetParameters();

Point<3, double> Triangulation_Shit_To_Local_Geometry(
    const Point<3, double> &p);

Point<3, double> Transform_4_to_5(const Point<3, double> &p);

Point<3, double> Transform_3_to_5(const Point<3, double> &p);

Point<3, double> Transform_2_to_5(const Point<3, double> &p);

Point<3, double> Transform_1_to_5(const Point<3, double> &p);

Point<3, double> Transform_0_to_5(const Point<3, double> &p);

inline bool file_exists(const std::string &name);

void PrepareStreams();

double Distance2D(Point<3, double>, Point<3, double> = Point<3, double>());

std::vector<types::global_dof_index> Add_Zero_Restraint(
    AffineConstraints<double> *, DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, IndexSet);

void add_vector_of_indices(IndexSet *, std::vector<types::global_dof_index>);

double hmax_for_cell_center(Point<3, double>);

double InterpolationPolynomial(double, double, double, double, double);

double InterpolationPolynomialDerivative(double, double, double, double,
                                         double);

double InterpolationPolynomialZeroDerivative(double, double, double);

double sigma(double, double, double);

#endif /* STATICFUNCTIONS_H_ */
