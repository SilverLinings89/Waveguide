#pragma once

/**
 * @file staticfunctions.h
 * @author Pascal Kraft
 * @brief This is an important file since it contains all the utility functions used anywhere in the code.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
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

/** @brief Legacy global defined in Main/main.cpp. Not used by the code in this folder tree. */
extern std::string solutionpath;
/** @brief Legacy global log stream defined in Main/main.cpp. Not used; the log is written by OutputManager. */
extern std::ofstream log_stream;
/** @brief Legacy global defined in Main/main.cpp (name of a log file for the constraints). Not used. */
extern std::string constraints_filename;
/** @brief Legacy global defined in Main/main.cpp (name of a log file for the assembly). Not used. */
extern std::string assemble_filename;
/** @brief Legacy global defined in Main/main.cpp (name of a log file for the preconditioner). Not used. */
extern std::string precondition_filename;
/** @brief Legacy global defined in Main/main.cpp (name of a log file for the solver). Not used. */
extern std::string solver_filename;
/** @brief Legacy global defined in Main/main.cpp (name of a log file for the complete run). Not used. */
extern std::string total_filename;
/** @brief Legacy global defined in Main/main.cpp. Not used. */
extern int StepsR;
/** @brief Legacy global defined in Main/main.cpp. Not used. */
extern int StepsPhi;
/** @brief Counter of the debugging function alert() in staticfunctions.cpp, which prints and increments it. */
extern int alert_counter;
/** @brief Legacy global declaration. Not used. */
extern std::string input_file_name;

using namespace dealii;

/**
 * @brief Runs a shell command and returns what it wrote to standard output.
 * Throws std::runtime_error if the pipe cannot be opened.
 *
 * @param cmd the command line, executed with popen().
 * @return std::string the output of the command.
 */
std::string exec(const char* cmd);

/**
 * @brief Returns the absolute path of a file or folder inside the repository (i.e. "Modes/output_mode.dat").
 * The repository root is taken from the environment variable WAVEGUIDE_ROOT if it is set and otherwise it is the
 * source directory the code was configured from. This makes the executables independent of the working directory.
 *
 * @param relative_path path relative to the repository root.
 * @return std::string the absolute path.
 */
std::string repository_path(const std::string & relative_path);

/**
 * @brief Entry of the tridiagonal matrix D of the Hardy space infinite elements.
 * D represents the derivative for the coefficient vector of a function in the Hardy space (see HSIEPolynomial,
 * which stores D and its inverse). All entries with |in_row - in_column| > 1 are zero.
 *
 * @param in_row row index, starting at 0.
 * @param in_column column index, starting at 0.
 * @param in_k0 the parameter kappa_0 of the infinite elements.
 * @return ComplexNumber the matrix entry.
 */
ComplexNumber matrixD(int in_row, int in_column, ComplexNumber in_k0);

/**
 * @brief Smallest and largest dof index in a list of interface dofs.
 *
 * @param in_data the interface dofs.
 * @return std::pair<DofNumber, DofNumber> first: the smallest index, second: the largest index.
 */
std::pair<DofNumber, DofNumber> get_max_and_min_dof_for_interface_data(std::vector<InterfaceDofData> in_data);

/**
 * @brief Strict ordering of positions: first by z, then by y, then by x. Coordinates z and y that differ by less than
 * 1e-4 count as equal. Dofs on two matching surfaces are sorted with this order to pair them.
 *
 * @param p1 first position.
 * @param p2 second position.
 * @return true if p1 comes before p2.
 */
bool comparePositions(Position p1, Position p2);

/**
 * @brief Compares two pairs of an index and a position by their positions with comparePositions().
 *
 * @param c1 first pair.
 * @param c2 second pair.
 * @return true if c1 comes before c2.
 */
bool compareDofBaseData(std::pair<DofNumber, Position> c1, std::pair<DofNumber, Position> c2);

/**
 * @brief Ordering of interface dofs: by base point (comparePositions()). Dofs with the same base point are ordered
 * by their member order and then by their index.
 * @return true if the first argument comes before the second.
 */
bool compareDofBaseDataAndOrientation(InterfaceDofData, InterfaceDofData);

/**
 * @brief Compares two surface cells by the centers of their surface faces with comparePositions().
 *
 * @param c1 first cell.
 * @param c2 second cell.
 * @return true if c1 comes before c2.
 */
bool compareSurfaceCellData(SurfaceCellData c1, SurfaceCellData c2);

/**
 * @brief Compares two interface dofs by their dof index.
 * @return true if the index of the first argument is smaller.
 */
bool compareDofDataByGlobalIndex(InterfaceDofData, InterfaceDofData);

/**
 * @brief Checks if the base points of two interface dofs coincide.
 *
 * @param a first dof.
 * @param b second dof.
 * @return true if the distance of the base points is smaller than FLOATING_PRECISION.
 */
bool areDofsClose(const InterfaceDofData &a, const InterfaceDofData &b);

/**
 * @brief Compares two field evaluations by their positions with comparePositions().
 *
 * @param field_a first evaluation.
 * @param field_b second evaluation.
 * @return true if field_a comes before field_b.
 */
bool compareFEAdjointEvals(const FEAdjointEvaluation field_a,const FEAdjointEvaluation field_b);

/**
 * @brief Prints the number of active cells of a mesh and how many boundary faces carry which boundary id.
 * The second argument is a name that is only printed; no file is written.
 */
void mesh_info(Triangulation<3> *, std::string);

/**
 * @brief Prints the number of active cells of a mesh and how many boundary faces carry which boundary id.
 * The definition in staticfunctions.cpp takes a reference and is not instantiated, so this overload cannot be called
 * from other files.
 */
template <int dim>
void mesh_info(const Triangulation<dim>);

/**
 * @brief Builds the Parameters object of the run.
 * Reads both parameter files, stores the MPI rank and the number of processes, applies the overrides and calls
 * Parameters::complete_data().
 *
 * @param run_file path of the run file (solver, parallelization, mesh size).
 * @param case_file path of the case file (geometry, materials, boundary method).
 * @param in_po overrides from the command line.
 * @return Parameters the completed parameter object.
 */
Parameters GetParameters(const std::string run_file,const std::string case_file, ParameterOverride& in_po);

/**
 * @brief Maps the vertices of the cube [-1,1]^3 to the corners of the local (subdomain) geometry.
 * A negative coordinate is mapped to the lower end of the local range of that direction and all others to the upper
 * end. Used by SquareMeshGenerator. Note that the z-direction is written to component 3 of the point instead of
 * component 2.
 *
 * @param p a vertex of the reference cube.
 * @return Position the mapped vertex.
 */
Position Triangulation_Shit_To_Local_Geometry(const Position &p);

/**
 * @name Coordinate changes between the boundary surfaces
 * The boundary ids are 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z. Transform_b_to_5 maps coordinates of a setup for
 * boundary b to the orientation of boundary 5 (+z) and Transform_5_to_b is its inverse, so code written for the
 * +z surface (surface mesh in the xy-plane, normal direction z) can be used for all six surfaces.
 */
///@{
/** @brief (x, y, z) -> (-x, y, z). */
Position Transform_4_to_5(const Position &p);

/** @brief (x, y, z) -> (x, -z, y). */
Position Transform_3_to_5(const Position &p);

/** @brief (x, y, z) -> (x, z, y). */
Position Transform_2_to_5(const Position &p);

/** @brief (x, y, z) -> (-z, y, x). */
Position Transform_1_to_5(const Position &p);

/** @brief (x, y, z) -> (z, y, x). */
Position Transform_0_to_5(const Position &p);

/** @brief (x, y, z) -> (-x, y, z). */
Position Transform_5_to_4(const Position &p);

/** @brief (x, y, z) -> (x, z, -y). */
Position Transform_5_to_3(const Position &p);

/** @brief (x, y, z) -> (x, z, y). */
Position Transform_5_to_2(const Position &p);

/** @brief (x, y, z) -> (z, y, -x). */
Position Transform_5_to_1(const Position &p);

/** @brief (x, y, z) -> (z, y, x). */
Position Transform_5_to_0(const Position &p);
///@}

/** @brief Legacy declaration without definition. Not used. */
std::vector<types::global_dof_index> Add_Zero_Restraint(
    AffineConstraints<double> *, DoFHandler<3>::active_cell_iterator &,
    unsigned int, unsigned int, unsigned int, bool, IndexSet);

/**
 * @brief Maximal cell diameter that the mesh generator requests for a cell with the given center: 0.05 inside the
 * waveguide core and 0.1 outside (micrometers). Used by SquareMeshGenerator. Note that the test for the height of the
 * core also uses the x-coordinate.
 */
double hmax_for_cell_center(Position);

/**
 * @brief Cubic Hermite interpolation on [0,1].
 * The arguments are the evaluation point z, the values at 0 and at 1 and the derivatives at 0 and at 1. For z < 0
 * the value at 0 is returned and for z > 1 the value at 1.
 */
double InterpolationPolynomial(double, double, double, double, double);

/**
 * @brief Derivative of InterpolationPolynomial() with respect to z; the arguments are the same.
 * For z < 0 the derivative at 0 is returned and for z > 1 the derivative at 1.
 */
double InterpolationPolynomialDerivative(double, double, double, double, double);

/**
 * @brief InterpolationPolynomial() with both end derivatives set to zero.
 * The arguments are the evaluation point z, the value at 0 and the value at 1.
 */
double InterpolationPolynomialZeroDerivative(double, double, double);

/**
 * @brief Linear ramp from 0 to 1.
 * The arguments are z, min and max. Returns 0 for z < min, 1 for z > max and (z - min) / (max - min) in between.
 * For min == max it is a step at min.
 */
double sigma(double, double, double);

/**
 * @brief Orientation of the edge from vertex_1 to vertex_2. It is used to give an edge the same direction on both
 * sides of an interface.
 * The edge is positive if the coordinate of its dominant direction increases from vertex_1 to vertex_2. Note that
 * the dominant direction is determined from vertex_2[i] - vertex_1[1] for all three i.
 *
 * @param vertex_1 start of the edge.
 * @param vertex_2 end of the edge.
 * @return true if the edge points in the positive coordinate direction.
 */
bool get_orientation(const Position &vertex_1, const Position &vertex_2);

/**
 * @brief Cross product u x v of two complex vectors with three components (no conjugation).
 *
 * @param u first vector.
 * @param v second vector.
 * @return NumericVectorLocal u x v.
 */
NumericVectorLocal crossproduct(const NumericVectorLocal &u,const NumericVectorLocal &v);

/**
 * @brief Writes a message to the console and to the log file if the logging level permits it.
 * The line is prefixed with the rank and the block indices of the process, "[rank:ix x iy x iz]", and the label.
 *
 * @param label origin of the message, usually "Class::function".
 * @param message the text.
 * @param level importance of the message, see is_visible_message_in_current_logging_level().
 */
void print_info(const std::string &label, const std::string &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
/** @brief Overload of print_info() that prints a number. */
void print_info(const std::string &label, const unsigned int message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
/** @brief Overload of print_info() that prints a list of numbers separated by spaces. */
void print_info(const std::string &label, const std::vector<unsigned int> &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);
/** @brief Overload of print_info() that prints six flags, one per boundary id, as "0 true 1 false ...". */
void print_info(const std::string &label, const std::array<bool,6> &message, LoggingLevel level = LoggingLevel::DEBUG_ONE);

/**
 * @brief Decides if a message of the given level is printed by this process.
 * If the configured level (GlobalParams.Logging_Level) is DEBUG_ONE or PRODUCTION_ONE, only process 0 prints.
 * Apart from that a message is printed if its level is not lower than the configured one in the order
 * DEBUG_ALL < DEBUG_ONE < PRODUCTION_ALL < PRODUCTION_ONE.
 *
 * @param level level of the message.
 * @return true if the message should be printed.
 */
bool is_visible_message_in_current_logging_level(LoggingLevel level = LoggingLevel::DEBUG_ONE);
/**
 * @brief Writes a message to the console and to the log of the OutputManager without checking the logging level.
 *
 * @param label origin of the message; printed in bold red.
 * @param message the text.
 */
void write_print_message(const std::string &label, const std::string &message);

/**
 * @brief The boundary on the opposite side of the domain (0 <-> 1, 2 <-> 3, 4 <-> 5).
 *
 * @param b_id a boundary id in [0,5].
 * @return BoundaryId the id of the opposing boundary.
 */
BoundaryId opposing_Boundary_Id(BoundaryId b_id);

/**
 * @brief Checks if two boundaries are on opposite sides of the domain.
 *
 * @param a first boundary id.
 * @param b second boundary id.
 * @return true if a and b are different and belong to the same coordinate direction.
 */
bool are_opposing_sites(BoundaryId a, BoundaryId b);

/**
 * @brief Pairs the dofs of two matching interfaces.
 * Both lists are sorted with compareDofBaseDataAndOrientation() (the arguments are modified) and then paired by
 * their position in the lists. The lists must have the same length.
 *
 * @param dofs_interface_1 dofs of the first interface.
 * @param dofs_interface_2 dofs of the second interface.
 * @return std::vector<DofCouplingInformation> one entry per pair with the two dof indices and the coupling value 1.
 */
std::vector<DofCouplingInformation> get_coupling_information(std::vector<InterfaceDofData> &dofs_interface_1, std::vector<InterfaceDofData> &dofs_interface_2);

/**
 * @brief Builds constraints that identify the dofs of two matching interfaces.
 * For every pair from get_coupling_information() the dof with the larger index is constrained to be equal to the one
 * with the smaller index. The returned object is not closed.
 *
 * @param dofs_interface_1 dofs of the first interface.
 * @param dofs_interface_2 dofs of the second interface.
 * @param max_dof number of dofs of the system; the constraints are defined on the index range [0, max_dof).
 * @return Constraints the constraint object.
 */
auto get_affine_constraints_for_InterfaceData(std::vector<InterfaceDofData> &dofs_interface_1, std::vector<InterfaceDofData> &dofs_interface_2, const unsigned int max_dof) -> Constraints;

/**
 * @brief Adds an offset to the dof indices of a list of interface dofs.
 *
 * @param dofs_interface_1 the list; it is modified in place.
 * @param shift the offset.
 */
void shift_interface_dof_data(std::vector<InterfaceDofData> * dofs_interface_1, unsigned int shift);

/**
 * @brief Creates a copy of a mesh in which the vertices are numbered in the order of comparePositions() (by z, then
 * y, then x).
 * With this numbering the meshes of neighboring subdomains and of the boundary methods have the same vertex order on
 * their common interfaces. Boundary ids of faces and lines are kept.
 *
 * @param original_triangulation the mesh to copy; only its coarse level is used.
 * @return dealii::Triangulation<3> the renumbered mesh.
 */
dealii::Triangulation<3> reforge_triangulation(dealii::Triangulation<3> * original_triangulation);

/**
 * @brief Classifies a surface type.
 *
 * @param in_st the surface type.
 * @return false for DIRICHLET_SURFACE and OPEN_SURFACE and true for all other types (also for NEIGHBOR_SURFACE).
 */
bool is_absorbing_boundary(SurfaceType in_st);

/**
 * @brief Decides which of two adjacent boundary surfaces owns the dofs on their common edge.
 * If both surfaces are no neighbor surfaces and have different types, self owns the dofs. If other is a neighbor
 * surface, self owns them if other has an odd id (upper side). In all other cases a fixed order of the boundary ids
 * decides.
 *
 * @param self the boundary that asks.
 * @param other the adjacent boundary.
 * @param in_level the sweeping level whose surface types are used.
 * @return true if the edge dofs belong to self.
 */
bool are_edge_dofs_locally_owned(BoundaryId self, BoundaryId other, unsigned int in_level);

/**
 * @brief The four boundaries that share an edge with a boundary.
 *
 * @param self a boundary id in [0,5].
 * @return std::vector<BoundaryId> all boundary ids except self and its opposing boundary, in ascending order.
 */
std::vector<BoundaryId> get_adjacent_boundary_ids(BoundaryId self);

/**
 * @brief The sweeping direction of a level of the hierarchy.
 * The highest level always sweeps in z. With two levels, level 1 sweeps in y; with three levels, level 1 sweeps in x
 * and level 2 in y.
 *
 * @param in_level the level, in [1, GlobalParams.Sweeping_Level].
 * @return SweepingDirection the direction.
 */
SweepingDirection get_sweeping_direction_for_level(unsigned int in_level);

/**
 * @brief Generates a pseudo random MPI tag from sender, receiver and level. Not used.
 *
 * @param global_rank_sender rank of the sender; it seeds the random number generator.
 * @param receiver rank of the receiver; the number of random numbers that are drawn.
 * @param level the sweeping level; the tag is smaller than the number of dofs of this level.
 * @return int the tag.
 */
int generate_tag(unsigned int global_rank_sender, unsigned int receiver, unsigned int level);

/**
 * @brief Splits a string at every occurrence of a delimiter.
 *
 * @param str the text.
 * @param token the delimiter.
 * @return std::vector<std::string> the parts without the delimiter. Empty parts are kept.
 */
std::vector<std::string> split(std::string str, std::string token);

/**
 * @brief Converts the name of an iterative solver to the enum value.
 *
 * @param in_name one of GMRES, MINRES, TFQMR, BICGS, CG and PCONLY.
 * @return SolverOptions the solver; GMRES for unknown names.
 */
SolverOptions solver_option(std::string in_name);

/**
 * @brief Packs field evaluations into a flat array of doubles for MPI communication.
 * Every evaluation uses nine numbers: the position (3) and real and imaginary part of the three components of
 * primal_field (6).
 *
 * @param inp the evaluations.
 * @return std::vector<double> array of size 9 * inp.size().
 */
std::vector<double> fe_evals_to_double(const std::vector<FEAdjointEvaluation>& inp);

/**
 * @brief Unpacks an array written by fe_evals_to_double() as data of the adjoint problem.
 * The positions are mirrored with adjoint_position_transformation() and the fields are transformed with
 * adjoint_field_transformation() and stored in the member adjoint_field. The other members are not set.
 *
 * @param inp array whose size is a multiple of 9.
 * @return std::vector<FEAdjointEvaluation> the evaluations.
 */
std::vector<FEAdjointEvaluation> fe_evals_from_double(const std::vector<double>& inp);

/**
 * @brief Mirrors a position in the z-direction at the center of the global domain: z -> z_max - (z - z_min).
 * The adjoint problem is solved on the mirrored geometry.
 *
 * @param in_p the position.
 * @return Position the mirrored position.
 */
Position adjoint_position_transformation(const Position in_p);

/**
 * @brief Transforms a field of the mirrored (adjoint) setup: the x- and z-components change their sign.
 *
 * @param in_field the field.
 * @return dealii::Tensor<1,3,ComplexNumber> (-E_x, E_y, -E_z).
 */
dealii::Tensor<1,3,ComplexNumber> adjoint_field_transformation(const dealii::Tensor<1,3,ComplexNumber> in_field);