
#ifndef STATICFUNCTIONS_H_
#define STATICFUNCTIONS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <fstream>
#include "../Core/Waveguide.h"
#include "../SpaceTransformations/SpaceTransformation.h"
using namespace dealii;

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

void set_the_st(SpaceTransformation *);

double dotproduct(Tensor<1, 3, double>, Tensor<1, 3, double>);

template <int dim>
void mesh_info(const parallel::distributed::Triangulation<dim>,
               const std::string);

template <int dim>
void mesh_info(const parallel::distributed::Triangulation<dim>);

double my_inter(double, double, double);

Parameters GetParameters();

Point<3, double> Triangulation_Stretch_X(const Point<3, double> &p);

Point<3, double> Triangulation_Stretch_Y(const Point<3, double> &p);

Point<3, double> Triangulation_Stretch_Z(const Point<3, double> &p);

inline bool file_exists(const std::string &name);

double TEMode00(dealii::Point<3, double>, int);

double sigma(double, double, double);

double InterpolationPolynomial(double, double, double, double, double);

double InterpolationPolynomialDerivative(double, double, double, double,
                                         double);

double InterpolationPolynomialZeroDerivative(double, double, double);

void PrepareStreams();

double Distance2D(Point<3, double>, Point<3, double> = Point<3, double>());

Point<3, double> Triangulation_Stretch_to_circle(const Point<3, double> &);

Point<3, double> Triangulation_Shift_Z(const Point<3, double> &);

Point<3, double> Triangulation_Stretch_Computational_Radius(
    const Point<3, double> &);

Point<3, double> Triangulation_Stretch_Computational_Rectangle(
    const Point<3, double> &);

Point<3, double> Triangulation_Transform_to_physical(const Point<3, double> &);

std::vector<types::global_dof_index> Add_Zero_Restraint(
    dealii::ConstraintMatrix *, dealii::DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, dealii::IndexSet);

std::vector<types::global_dof_index> Add_Zero_Restraint_test(
    dealii::ConstraintMatrix *, dealii::DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, dealii::IndexSet);

void add_vector_of_indices(dealii::IndexSet *,
                           std::vector<types::global_dof_index>);

#endif /* STATICFUNCTIONS_H_ */
