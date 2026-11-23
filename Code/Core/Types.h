#pragma once

/**
 * @file Types.h
 * @author Pascal Kraft
 * @brief This file contains all type declarations used in this project.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <string>
#include <utility>
#include <array>
#include <vector>
#include <complex>
#include <deal.II/base/point.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h> 
#include <deal.II/base/index_set.h>
#include "../BoundaryCondition/DofData.h"

struct RayAngelingData;
class BounadaryCondition;

/** @brief One complex component of the electric field. */
using EFieldComponent          = std::complex<double>;
/** @brief The three complex components (x, y, z) of the electric field at a point. */
using EFieldValue              = std::array<EFieldComponent, 3>;
/** @brief A number of degrees of freedom. */
using DofCount                 = unsigned int;
/** @brief A point in three dimensional space. */
using Position                 = dealii::Point<3, double>;
/** @brief A point in two dimensions, e.g. on a surface mesh or in the cross-section of the waveguide. */
using Position2D               = dealii::Point<2, double>;
/** @brief The index of a degree of freedom. Whether the numbering is local to a domain or global on a sweeping level depends on the context. */
using DofNumber                = unsigned int;
/** @brief A dof index together with a position that can be used to sort dofs geometrically. */
using DofSortingData           = std::pair<DofNumber, Position>;
/** @brief A serial complex vector. Used for data in the local numbering of one domain (inner domain or boundary condition). */
using NumericVectorLocal       = dealii::Vector<EFieldComponent>;
/** @brief A distributed PETSc vector. Used for solutions and right-hand sides in the global numbering of a sweeping level. */
using NumericVectorDistributed = dealii::PETScWrappers::MPI::Vector;
/** @brief A distributed sparse PETSc matrix with complex entries. */
using SparseComplexMatrix      = dealii::PETScWrappers::MPI::SparseMatrix;
/** @brief Index of a level in the sweeping hierarchy. Level 0 is the level of the local problems with direct solvers. */
using SweepingLevel            = unsigned int;
/** @brief Polynomial order of a Hardy space infinite element in the infinite direction. */
using HSIEElementOrder         = unsigned int;
/** @brief Order of the Nedelec elements. */
using NedelecElementOrder      = unsigned int;
/** @brief Id of one of the six surfaces of a cuboid domain: 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z. */
using BoundaryId               = unsigned int;
/** @brief Complex number type used for all field values and matrix entries. */
using ComplexNumber            = std::complex<double>;
/** @brief Dof handler on a two dimensional (surface) mesh. */
using DofHandler2D             = dealii::DoFHandler<2>;
/** @brief Dof handler on a three dimensional mesh. */
using DofHandler3D             = dealii::DoFHandler<3>;
/** @brief Iterator over the active cells of a DofHandler2D. */
using CellIterator2D           = DofHandler2D::active_cell_iterator;
/** @brief Iterator over the active cells of a DofHandler3D. */
using CellIterator3D           = DofHandler3D::active_cell_iterator;
/** @brief List of dof descriptions of a Hardy space infinite element surface, see DofData. */
using DofDataVector            = std::vector<DofData>;
/** @brief Symbolic expression (SymEngine). Used to differentiate coordinate transformations symbolically. */
using MathExpression           = dealii::Differentiation::SD::Expression;
/** @brief A three dimensional triangulation. */
using Mesh                     = dealii::Triangulation<3>;
/** @brief Complex 3x3 tensor. Used for material tensors such as the permittivity and for the tensors of transformation optics and PML. */
using MaterialTensor           = dealii::Tensor<2,3,ComplexNumber>;
/** @brief Angling data of the four infinite rays that start at the vertices of one surface cell, see RayAngelingData. */
using FaceAngelingData         = std::array<RayAngelingData, 4>;
/** @brief One flag per surface of a cuboid domain, indexed by BoundaryId. True if the domain is truncated at that surface. */
using CubeSurfaceTruncationState = std::array<bool, 6>;
/** @brief Complex values associated with a list of dofs. Not used in the current code. */
using DofFieldTrace            = std::vector<ComplexNumber>;
/** @brief Affine constraints with complex coefficients (hanging nodes are not used; this holds Dirichlet values and couplings between dofs). */
using Constraints               = dealii::AffineConstraints<ComplexNumber>;
/** @brief A list of dof indices. */
using DofIndexVector           = std::vector<DofNumber>;

/** @brief Tolerance for comparisons of floating point values, mostly of coordinates. */
const double FLOATING_PRECISION = 0.00001;

/**
 * @brief Maps the edges of a surface mesh to the surfaces of the cuboid they are shared with.
 *
 * The first index is the BoundaryId of the surface the two dimensional mesh belongs to. The second index names an edge of that mesh in its own two dimensional coordinates: 0 = lower first coordinate, 1 = upper first coordinate, 2 = lower second coordinate, 3 = upper second coordinate. The value is the BoundaryId of the neighboring surface of the cuboid at that edge. It is used as boundary id of the faces on that edge of the surface mesh.
 */
const std::vector<std::vector<unsigned int>>  edge_to_boundary_id = {
    {4,5,2,3}, {5,4,2,3}, {0,1,4,5}, {0,1,5,4}, {1,0,2,3}, {0,1,2,3}
};

/**
 * @brief Data of one part of a system matrix stored in serial data structures.
 *
 * This struct is not used in the current code.
 */
struct LocalMatrixPart {
  /** @brief Constraints for the dofs of this part. */
  dealii::AffineConstraints<ComplexNumber> constraints; // prepped
  /** @brief Sparsity pattern of matrix. */
  dealii::SparsityPattern sp;
  /** @brief The serial matrix of this part. */
  dealii::SparseMatrix<ComplexNumber> matrix;
  /** @brief Number of dofs of this part. */
  unsigned int n_dofs; // done
  /** @brief Dofs on the interface on the lower side in sweeping direction. */
  dealii::IndexSet lower_sweeping_dofs; //done
  /** @brief Dofs on the interface on the upper side in sweeping direction. */
  dealii::IndexSet upper_sweeping_dofs; //done
  /** @brief The dofs that belong to this part. */
  dealii::IndexSet local_dofs; // done
};

/** @brief Describes for one edge of a surface cell if the infinite direction of the Hardy space elements is tilted against the surface normal. */
struct EdgeAngelingData {
  /** @brief Index of the edge. */
  unsigned int edge_index;
  /** @brief True if the infinite direction is tilted in the first surface coordinate. */
  bool angled_in_x = false;
  /** @brief True if the infinite direction is tilted in the second surface coordinate. */
  bool angled_in_y = false;
};

/** @brief Describes for one vertex of a surface cell if the infinite direction of the Hardy space elements is tilted against the surface normal. */
struct VertexAngelingData {
  /** @brief Index of the vertex. */
  unsigned int vertex_index;
  /** @brief True if the infinite direction is tilted in the first surface coordinate. */
  bool angled_in_x = false;
  /** @brief True if the infinite direction is tilted in the second surface coordinate. */
  bool angled_in_y = false;
};

/** @brief Angling data of one edge and one vertex of a surface cell. Not used in the current code. */
struct CellAngelingData {
  /** @brief Angling data of the edge. */
  EdgeAngelingData edge_data; 
  /** @brief Angling data of the vertex. */
  VertexAngelingData vertex_data;
};

/** @brief Ownership information of a dof. Not used in the current code. */
struct DofOwner {
  /** @brief Index of the process that owns the dof. */
  unsigned int owner = 0;
  /** @brief True if the dof belongs to a boundary condition and not to the inner domain. */
  bool is_boundary_dof = false;
  /** @brief BoundaryId of the surface the dof belongs to if is_boundary_dof is true. */
  unsigned int surface_id = 0;
};

/**
 * @brief Shape of the ramp used to fade the input signal in along z (see ExactSolutionRamped).
 *
 * C1: cubic ramp with continuous first derivative. C0: linear ramp, only continuous.
 */
enum SignalTaperingType {
  C1, C0
};

/**
 * @brief How the input signal enters the system.
 *
 * Tapering: the mode is faded in by a volume source term between the parameters tapering_min_z and tapering_max_z and the input surface gets an absorbing boundary condition. Dirichlet: the mode is prescribed as Dirichlet values on the input surface (boundary id 4) of the processes with z index 0.
 */
enum SignalCouplingMethod {
  Tapering, Dirichlet
};

/**
 * @brief Kinds of output files the FileLogger distinguishes.
 *
 * ConvergenceCSV: convergence history as CSV. ParaviewVTU: field output for ParaView. TexReport: report in LaTeX format. MetaText: plain text file with meta data of a run.
 */
enum FileType {
  ConvergenceCSV, ParaviewVTU, TexReport, MetaText
};

/**
 * @brief Kinds of entries that can be sent to a logger. Not used in the current code.
 *
 * ConvergenceHistoryEntry: residual of one step of an iterative solver. FinalConvergenceStep: last step of a solver run. SolverMetaData: general data about a solver.
 */
enum LoggerEntryType {
  ConvergenceHistoryEntry, FinalConvergenceStep, SolverMetaData, 
};

/** @brief Additional data used by the FileLogger to build a file name. */
struct FileMetaData {
  /** @brief Level in the sweeping hierarchy the file belongs to. */
  unsigned int hsie_level;
};

/**
 * @brief Severity of a log message and at the same time the threshold set in the parameters (see print_info()).
 *
 * A message is printed if its level is greater than or equal to the configured level in the order DEBUG_ALL < DEBUG_ONE < PRODUCTION_ALL < PRODUCTION_ONE. If the configured level is DEBUG_ONE or PRODUCTION_ONE, only the process with MPI rank 0 prints. With the configured levels DEBUG_ALL and PRODUCTION_ALL all processes print.
 */
enum LoggingLevel {
  DEBUG_ALL, DEBUG_ONE, PRODUCTION_ALL, PRODUCTION_ONE
};

/** @brief Describes the direction of one infinite ray of a Hardy space infinite element, which starts at a vertex of a surface cell. */
struct RayAngelingData {
  /** @brief True if the ray is tilted against the surface normal in the first surface coordinate. */
  bool is_x_angled = false;
  /** @brief True if the ray is tilted against the surface normal in the second surface coordinate. */
  bool is_y_angled = false;
  /** @brief Position of the vertex the ray starts at, in the coordinates of the surface mesh. */
  Position2D position_of_base_point;
};

/** @brief Describes how a coordinate of a surface is related to the coordinates of the inner domain. Not used in the current code. */
struct BoundaryInformation {
  /**
   * @brief Stores both arguments in the members.
   *
   * @param in_coord Value for inner_coordinate.
   * @param neg Value for negate_value.
   */
  BoundaryInformation(unsigned int in_coord, bool neg) {
    inner_coordinate = in_coord;
    negate_value = neg;
  }

  /** @brief Index (0, 1 or 2) of the coordinate of the inner domain. */
  unsigned int inner_coordinate;
  /** @brief True if the value has to be multiplied by -1. */
  bool negate_value;
};

/** @brief Describes that two dofs on two matching interfaces represent the same function, up to a factor. */
struct DofCouplingInformation {
  /** @brief Index of the dof on the first interface. */
  DofNumber first_dof;
  /** @brief Index of the dof on the second interface. */
  DofNumber second_dof;
  /** @brief Factor between the two dofs. Currently always 1. */
  double coupling_value;
};

/**
 * @brief A dof on a surface together with the data required to match it with the same dof seen from another domain.
 *
 * Two domains that share a surface (inner domain and boundary condition, or two boundary conditions) both build a vector of these objects and sort it with compareDofBaseDataAndOrientation(), i.e. by base_point, then order, then index. The i-th entries of both vectors then describe the same dof.
 */
struct InterfaceDofData {
  /** @brief Sets the index to 0 and the base point to the origin. The order is not initialized. */
  InterfaceDofData() {
    index = 0;
    base_point = { 0, 0, 0 };
  }

  /**
   * @brief Sets index and base point. The order is not initialized.
   *
   * @param in_index Index of the dof.
   * @param in_position Base point of the dof.
   */
  InterfaceDofData(const DofNumber &in_index, const Position &in_position) {
    index = in_index;
    base_point = in_position;
  }

  /** @brief Index of the dof in the numbering of the domain that created this object. */
  DofNumber index;
  /** @brief Center of the edge or face (or position of the vertex) the dof is associated with. */
  Position base_point;
  /** @brief Distinguishes several dofs with the same base point, e.g. the index of the dof on its edge. */
  unsigned int order;
};

/** @brief Describes which mesh entity a dof of a Hardy space infinite element surface belongs to. Not used in the current code. */
struct DofAssociation {
  /** @brief True if the dof is associated with an edge of the surface mesh, false if it is associated with a face. */
  bool is_edge;
  /** @brief Index of the edge if is_edge is true. */
  DofNumber edge_index;
  /** @brief Id string of the surface cell if is_edge is false. */
  std::string face_index;
  /** @brief Index of the dof in the numbering of the surface. */
  DofNumber dof_index_on_hsie_surface;
  /** @brief Center of the edge or face the dof is associated with. */
  Position base_point;
  /** @brief False if the orientation of the edge is reversed compared to the reference orientation. */
  bool true_orientation;
};

/**
 * @brief The Jacobian of the mapping of a Hardy space infinite element at one point and the two tensors derived from it.
 *
 * Computed by JacobianForCell::get_C_G_and_J().
 */
struct JacobianAndTensorData {
  /** @brief \f$C = J^T J / |\det J|\f$. */
  dealii::Tensor<2,3,double> C;
  /** @brief \f$G = |\det J| \, J^{-1} J^{-T}\f$. */
  dealii::Tensor<2,3,double> G;
  /** @brief The Jacobian \f$J\f$ of the mapping. */
  dealii::Tensor<2,3,double> J;
};

/** @brief Dof counts of a Hardy space infinite element surface for one kind of mesh entity (vertices, edges or faces). */
struct DofCountsStruct {
  /** @brief Number of dofs that only exist because of the infinite elements. */
  unsigned int hsie     = 0;
  /** @brief Number of dofs that are also dofs of the inner domain on the surface. */
  unsigned int non_hsie = 0;
  /** @brief Sum of hsie and non_hsie. */
  unsigned int total    = 0;
};

/** @brief Dof counts and index sets of one process on one sweeping level. Not used in the current code. */
struct LevelDofOwnershipData {
  /** @brief Number of dofs of the level on all processes. Size of the index sets. */
  unsigned int global_dofs;
  /** @brief Number of dofs owned by this process. */
  unsigned int owned_dofs;
  /** @brief The dofs owned by this process. */
  dealii:: IndexSet locally_owned_dofs;
  /** @brief The dofs on the input interface of this process. */
  dealii:: IndexSet input_dofs;
  /** @brief The dofs on the output interface of this process. */
  dealii:: IndexSet output_dofs;
  /** @brief The dofs this process needs access to: owned dofs and dofs of neighbors that couple with them. */
  dealii:: IndexSet locally_relevant_dofs;

  /** @brief Sets both counts to zero and clears all index sets. */
  LevelDofOwnershipData() {
    global_dofs = 0;
    owned_dofs  = 0;
    locally_owned_dofs.clear();
    input_dofs.clear();
    output_dofs.clear();
    locally_relevant_dofs.clear();
  }

  /**
   * @brief Sets the number of global dofs and creates empty index sets of that size.
   *
   * @param in_global Number of dofs of the level on all processes.
   */
  LevelDofOwnershipData(unsigned int in_global) {
    global_dofs = in_global;
    owned_dofs  = 0;
    locally_owned_dofs.clear();
    locally_owned_dofs.set_size(in_global);
    input_dofs.clear();
    input_dofs.set_size(in_global);
    output_dofs.clear();
    output_dofs.set_size(in_global);
    locally_relevant_dofs.clear();
    locally_relevant_dofs.set_size(in_global);
  }
};

/** @brief A pair of dofs that are coupled by a constraint. */
struct ConstraintPair {
  /** @brief Index of the first dof. Used as the sorting key by compareConstraintPairs(). */
  unsigned int left;
  /** @brief Index of the second dof. */
  unsigned int right;
  /** @brief Sign of the coupling between the two dofs. */
  bool sign;
};

/** @brief The dofs of one cell face on a surface together with the center of that face, which is used to sort these objects (see compareSurfaceCellData()). */
struct SurfaceCellData {
  /** @brief Indices of the dofs on the face. */
  std::vector<DofNumber> dof_numbers;
  /** @brief Center of the face. */
  Position surface_face_center;
};

/** @brief A named sequence of values. Used for the residual history of one run of an iterative solver (see ResidualOutputGenerator). */
struct DataSeries {
  /** @brief The values in the order they were added, i.e. one residual per solver step. */
  std::vector<double> values;
  /** @brief True once the solver run has finished and no more values will be added. */
  bool is_closed;
  /** @brief Label of the series in outputs. */
  std::string name;
};

/** @brief Errors of a finite element solution compared to the exact solution, as computed by InnerDomain::compute_errors(). */
struct FEErrorStruct {
  /** @brief Error in the \f$L^2\f$ norm. */
  double L2 = 0;
  /** @brief Error in the \f$L^\infty\f$ norm. */
  double Linfty = 0;
};

/** @brief Primal and adjoint fields and their curls at one quadrature point. Computed by InnerDomain::compute_local_shape_gradient_data(). */
struct FEAdjointEvaluation {
  /** @brief The quadrature point in the coordinates of the mesh. */
  Position x;
  /** @brief Solution of the primal problem at x. */
  dealii::Tensor<1,3,ComplexNumber> primal_field;
  /** @brief Solution of the adjoint problem at x. */
  dealii::Tensor<1,3,ComplexNumber> adjoint_field;
  /** @brief Curl of the primal solution at x. */
  dealii::Tensor<1,3,ComplexNumber> primal_field_curl;
  /** @brief Curl of the adjoint solution at x. */
  dealii::Tensor<1,3,ComplexNumber> adjoint_field_curl;
};

/**
 * @brief Values and derivatives of the mode profile at one point of the cross-section.
 *
 * Computed by ExactSolution::get_derivative_terms() with finite differences on the grid of the tabulated mode data. f is the x-component of the mode profile and h is its z-component multiplied with the scaling factor of the z-component.
 */
struct J_derivative_terms {
  /** @brief Value of f at the point. */
  ComplexNumber f, d_f_dyy, d_f_dxy, d_f_dx;
  /** @var d_f_dyy
   * @brief Second derivative of f in y. */
  /** @var d_f_dxy
   * @brief Mixed second derivative of f in x and y. */
  /** @var d_f_dx
   * @brief First derivative of f in x. */
  /** @brief Value of h at the point. */
  ComplexNumber h, d_h_dx, d_h_dy, d_h_dxx, d_h_dyy;
  /** @var d_h_dx
   * @brief First derivative of h in x. */
  /** @var d_h_dy
   * @brief First derivative of h in y. */
  /** @var d_h_dxx
   * @brief Second derivative of h in x. */
  /** @var d_h_dyy
   * @brief Second derivative of h in y. */
  /** @brief Propagation constant of the mode in z-direction. */
  double beta;
};