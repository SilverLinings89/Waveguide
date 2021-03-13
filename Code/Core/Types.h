#pragma once

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

using EFieldComponent          = std::complex<double>;
using EFieldValue              = std::array<EFieldComponent, 3>;
using DofCount                 = unsigned int;
using Position                 = dealii::Point<3, double>;
using Position2D               = dealii::Point<2, double>;
using DofNumber                = unsigned int;
using DofSortingData           = std::pair<DofNumber, Position>;
using NumericVectorLocal       = dealii::Vector<EFieldComponent>;
using NumericVectorDistributed = dealii::PETScWrappers::MPI::Vector;
using SparseComplexMatrix      = dealii::PETScWrappers::MPI::SparseMatrix;
using SweepingLevel            = unsigned int;
using HSIEElementOrder         = unsigned int;
using NedelecElementOrder      = unsigned int;
using BoundaryId               = unsigned int;
using ComplexNumber            = std::complex<double>;
using DofHandler2D             = dealii::DoFHandler<2>;
using DofHandler3D             = dealii::DoFHandler<3>;
using CellIterator2D           = DofHandler2D::active_cell_iterator;
using CellIterator3D           = DofHandler3D::active_cell_iterator;
using DofDataVector            = std::vector<DofData>;
using MathExpression           = dealii::Differentiation::SD::Expression;
using Mesh                     = dealii::Triangulation<3>;
using MaterialTensor           = dealii::Tensor<2,3,ComplexNumber>;
using FaceAngelingData         = std::array<RayAngelingData, 4>;

struct EdgeAngelingData {
  unsigned int edge_index;
  bool angled_in_x = false;
  bool angled_in_y = false;
};

struct VertexAngelingData {
  unsigned int vertex_index;
  bool angled_in_x = false;
  bool angled_in_y = false;
};

struct CellAngelingData {
  EdgeAngelingData edge_data; 
  VertexAngelingData vertex_data;
};

enum SignalTaperingType {
  C1, C0
};

enum FileType {
  ConvergenceCSV, ParaviewVTU, TexReport, MetaText
};

enum LoggerEntryType {
  ConvergenceHistoryEntry, FinalConvergenceStep, SolverMetaData, 
};

struct FileMetaData {
  unsigned int hsie_level;
};

enum LoggingLevel {
  DEBUG_ALL, DEBUG_ONE, PRODUCTION_ALL, PRODUCTION_ONE
};

struct RayAngelingData {
  bool is_x_angled = false;
  bool is_y_angled = false;
  Position2D position_of_base_point;
};

struct BoundaryInformation {
  BoundaryInformation(unsigned int in_coord, bool neg) {
    inner_coordinate = in_coord;
    negate_value = neg;
  }

  unsigned int inner_coordinate;
  bool negate_value;
};

struct DofCouplingInformation {
  DofNumber first_dof;
  DofNumber second_dof;
  double coupling_value;
};

struct InterfaceDofData {
  InterfaceDofData() {
    index = 0;
    base_point = { 0, 0, 0 };
  }

  InterfaceDofData(const DofNumber &in_index, const Position &in_position) {
    index = in_index;
    base_point = in_position;
  }

  DofNumber index;
  Position base_point;
  unsigned int order;
};

struct DofAssociation {
  bool is_edge;
  DofNumber edge_index;
  std::string face_index;
  DofNumber dof_index_on_hsie_surface;
  Position base_point;
  bool true_orientation;
};

struct JacobianAndTensorData {
  dealii::Tensor<2,3,double> C;
  dealii::Tensor<2,3,double> G;
  dealii::Tensor<2,3,double> J;
};

struct DofCountsStruct {
  unsigned int hsie     = 0;
  unsigned int non_hsie = 0;
  unsigned int total    = 0;
};

struct LevelDofOwnershipData {
  unsigned int global_dofs;
  unsigned int owned_dofs;
  dealii:: IndexSet locally_owned_dofs;
  dealii:: IndexSet input_dofs;
  dealii:: IndexSet output_dofs;
  dealii:: IndexSet locally_relevant_dofs;

  LevelDofOwnershipData() {
    global_dofs = 0;
    owned_dofs  = 0;
    locally_owned_dofs.clear();
    input_dofs.clear();
    output_dofs.clear();
    locally_relevant_dofs.clear();
  }

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

struct ConstraintPair {
  unsigned int left;
  unsigned int right;
  bool sign;
};
